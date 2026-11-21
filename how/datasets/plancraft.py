import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import imageio
import networkx as nx
import plancraft
from plancraft.config import PlancraftExample
from plancraft.environment.env import PlancraftEnvironment
from plancraft.environment.planner import get_subplans
from plancraft.environment.recipes import RECIPES, BaseRecipe
from plancraft.environment.search import gold_search_recipe
from smolagents import Tool
from functools import cache

import wandb

from pydantic import BaseModel, field_validator, model_validator


MAX_PLAN_COST = 30


@cache
def convert_to_slot_index(slot: str) -> int:
    slot = slot.strip()
    grid_map = {
        "0": 0,
        "A1": 1,
        "A2": 2,
        "A3": 3,
        "B1": 4,
        "B2": 5,
        "B3": 6,
        "C1": 7,
        "C2": 8,
        "C3": 9,
    }
    if slot in grid_map:
        return grid_map[slot]
    else:
        # check if it is an inventory slot
        if len(slot) < 2 or len(slot) > 3 or slot[0] != "I" or slot[1] == "0":
            raise ValueError("Invalid slot index")
        # convert I1 to 10, I2 to 11, I3 to 12, ..., I36 to 45
        if slot[1:].isdigit() and 1 <= int(slot[1:]) <= 36:
            return int(slot[1:]) + 9
        raise ValueError("Invalid slot index")


@cache
def convert_from_slot_index(slot_index: int) -> str:
    grid_map = {
        0: "0",
        1: "A1",
        2: "A2",
        3: "A3",
        4: "B1",
        5: "B2",
        6: "B3",
        7: "C1",
        8: "C2",
        9: "C3",
    }
    if slot_index < 10:
        return grid_map[slot_index]
    elif slot_index < 46:
        return f"I{slot_index - 9}"
    raise ValueError("Invalid slot index")


class MoveAction(BaseModel):
    """ "Moves an item from one slot to another"""

    slot_from: int
    slot_to: int
    quantity: int
    action_type: str = "move"

    @field_validator("action_type", mode="before")
    def fix_action_type(cls, value) -> str:
        return "move"

    @field_validator("slot_from", "slot_to", mode="before")
    def transform_str_to_int(cls, value) -> int:
        # if value is a string like A1 or I1, convert it to an integer
        if isinstance(value, str):
            try:
                return convert_to_slot_index(value)
            except ValueError:
                raise AttributeError(
                    "Source and Target must be 0 or A1 to C3 or I1 to I36"
                )
        return value

    @field_validator("quantity", mode="before")
    def transform_quantity(cls, value) -> int:
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise AttributeError("quantity must be an integer")
        return value

    @model_validator(mode="after")
    def validate(self):
        if self.slot_from == self.slot_to:
            raise AttributeError("Source and Target must be different")
        if self.slot_from < 0 or self.slot_from > 45:
            raise AttributeError("Source must be 0 or A1 to C3 or I1 to I36")
        if self.slot_to == 0:
            raise AttributeError("You cannot move items into 0")
        if self.slot_to < 1 or self.slot_to > 45:
            raise AttributeError("Target must be A1 to C3 or I1 to I36")
        if self.quantity < 1 or self.quantity > 64:
            raise AttributeError("quantity must be between 1 and 64")
        return self

    def __str__(self):
        return f"move: from {convert_from_slot_index(self.slot_from)} to {convert_from_slot_index(self.slot_to)} with quantity {self.quantity}"


class SmeltAction(BaseModel):
    """Smelts an item and moves the result into a new slot"""

    slot_from: int
    slot_to: int
    quantity: int
    action_type: str = "smelt"

    @field_validator("action_type", mode="before")
    def fix_action_type(cls, value) -> str:
        return "smelt"

    @field_validator("slot_from", "slot_to", mode="before")
    def transform_str_to_int(cls, value) -> int:
        # if value is a string like [A1] or [I1], convert it to an integer
        if isinstance(value, str):
            try:
                return convert_to_slot_index(value)
            except ValueError:
                raise AttributeError(
                    "Source and Target must be 0 or A1 to C3 or I1 to I36"
                )
        return value

    @field_validator("quantity", mode="before")
    def transform_quantity(cls, value) -> int:
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise AttributeError("quantity must be an integer")
        return value

    @model_validator(mode="after")
    def validate(self):
        if self.slot_from == self.slot_to:
            raise AttributeError("Source and Target must be different")
        if self.slot_from < 0 or self.slot_from > 45:
            raise AttributeError("Source must be 0 or A1 to C3 or I1 to I36")
        if self.slot_to == 0:
            raise AttributeError("You cannot smelt items into 0")
        if self.slot_to < 1 or self.slot_to > 45:
            raise AttributeError("Target must be A1 to C3 or I1 to I36")
        if self.quantity < 1 or self.quantity > 64:
            raise AttributeError("quantity must be between 1 and 64")
        return self

    def __str__(self):
        return f"smelt: from {convert_from_slot_index(self.slot_from)} to {convert_from_slot_index(self.slot_to)} with quantity {self.quantity}"


class StopAction(BaseModel):
    """
    Action that model can take to stop planning - decide impossible to continue
    Note: also known as the "impossible" action
    """

    reason: str = ""

    def __str__(self):
        return f"impossible: {self.reason}"


# when symbolic action is true, can either move objects around or smelt
SymbolicAction = MoveAction | SmeltAction


def get_objective_str(target: str) -> str:
    return f"Craft an item of type: {target}"


def target_and_inventory_to_text_obs(target: str, inventory: dict) -> str:
    """
    Convert inventory dict to text observation
    """
    objective = get_objective_str(target)
    inventory_str = ""
    # sort by slot number
    for slot, item in sorted(inventory.items(), key=lambda item: item[0]):
        # skip items with quantity 0
        if item["quantity"] <= 0:
            continue
        slot_str = convert_from_slot_index(slot)
        inventory_str += f"\n - {item['type']} {slot_str} quantity {item['quantity']}"
    return f"{objective}\ninventory:{inventory_str}"


def get_topological_order():
    """
    Get the topological order of the recipes
    """
    RECIPE_GRAPH = nx.DiGraph()
    for _, recipes in RECIPES.items():
        for recipe in recipes:
            RECIPE_GRAPH.add_node(recipe.result.item)
            for ingredient in recipe.inputs:
                RECIPE_GRAPH.add_node(ingredient)
                RECIPE_GRAPH.add_edge(ingredient, recipe.result.item)

    # Detect and break cycles
    for cycle in nx.simple_cycles(RECIPE_GRAPH):
        if len(cycle) > 1:
            # Remove one edge in the cycle
            u, v = cycle[0], cycle[1]
            if RECIPE_GRAPH.has_edge(u, v):
                RECIPE_GRAPH.remove_edge(u, v)
    return list(nx.topological_sort(RECIPE_GRAPH))


class PlancraftToolsEvaluator:
    def __init__(
        self,
        run_name: str,
        output_dir: str = "output",
        split: str = "val.small",
        max_steps: int = 30,
        resume: bool = False,
        use_multimodal_content_format: bool = False,
        use_text_inventory: bool = True,
        resolution: str = "high",
        curriculum: bool = False,
        use_images: bool = False,
    ):
        self.run_name = run_name
        self.output_dir = f"{output_dir}/{run_name}/{split}"
        self.max_steps = max_steps
        self.resume = resume
        self.generation_number = 0
        self.curriculum = curriculum

        self.use_multimodal_content_format = use_multimodal_content_format
        self.use_text_inventory = use_text_inventory
        self.resolution = resolution
        self.use_images = use_images  # whether to pass images in message content

        # load examples
        self.examples: list[PlancraftExample] = self.load_dataset(split)

    def save_results_dict(self, example: PlancraftExample, results_dict: dict):
        output_dir = f"{self.output_dir}/{self.generation_number}"
        os.makedirs(output_dir, exist_ok=True)
        json_path = f"{output_dir}/{example.id}.json"
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        if wandb.run is not None:
            wandb.save(json_path, policy="now")

    def save_images(self, example: PlancraftExample, frames: list):
        if len(frames) == 0:
            return
        output_dir = f"{self.output_dir}/{self.generation_number}"
        os.makedirs(output_dir, exist_ok=True)
        imageio.mimsave(f"{output_dir}/{example.id}.gif", frames)
        # upload to wandb
        if wandb.run is not None:
            wandb.save(f"{output_dir}/{example.id}.gif", policy="now")

    def load_results_dict(self, example: PlancraftExample) -> Optional[dict]:
        path = f"{self.output_dir}/{self.generation_number}/{example.id}.json"
        if not os.path.exists(path) or not self.resume:
            return None
        with open(path, "r") as f:
            return json.load(f)

    def load_dataset(self, dataset_split: str) -> list[PlancraftExample]:
        # load the dataset from the json file
        if dataset_split in ["train", "val", "test", "val.small"]:
            dataset_file = (
                Path(plancraft.__file__).parent / "data" / f"{dataset_split}.json"
            )
        else:
            # if the split is not one of the predefined ones,
            #  assume it's a custom dataset located in the data directory
            dataset_file = Path(__file__).parent / "data" / f"{dataset_split}.json"

        with open(dataset_file, "r") as f:
            dataset = json.load(f)
            examples = [PlancraftExample(**example) for example in dataset]

        if self.curriculum:
            order = get_topological_order()
            # sort the examples by complexity
            examples = sorted(
                examples,
                key=lambda x: order.index(x.target),
            )
        return examples

    def check_done(self, inventory: dict, target: str):
        """
        Check that target object is obtained
        """
        for slot, item in inventory.items():
            # ensure the target is in the inventory (not in slot 0)
            if target == item["type"] and slot != 0:
                return True
        return False

    def expand_observation(
        self,
        target: str,
        observation: dict,
    ) -> None:
        """
        Adds the target and inventory to the observation.
        This is used to create the message content for the model.
        """
        observation["target"] = target

        if not self.use_text_inventory:
            text_content = get_objective_str(observation["target"])
        else:
            # if not multimodal, we only have text - we format the inventory as text
            text_content = target_and_inventory_to_text_obs(
                observation["target"], observation["inventory"]
            )

        if not self.use_multimodal_content_format:
            # if not multimodal, we have a single content type
            observation["message"] = text_content
        else:
            # if multimodal, we have a list of content types
            content_list = [{"type": "text", "text": text_content}]
            if self.use_images:
                content_list.append({"type": "image"})
            observation["message"] = {"content": content_list}

    def eval_example(
        self,
        example: PlancraftExample,
        model,
        terminal_ui,
    ) -> dict:
        if results := self.load_results_dict(example):
            # if we have already processed this example, skip it
            return results

        frames, actions, num_actions_to_goal = [], [], []

        model.reset()
        environment = PlancraftEnvironment(
            inventory=deepcopy(example.slotted_inventory),
            resolution=self.resolution,
        )

        observation = environment.step()
        frames.append(observation["image"])
        self.expand_observation(example.target, observation)

        num_actions = get_num_actions_to_goal(
            observation,
            impossible=example.impossible,
        )
        num_actions_to_goal.append(num_actions)

        # Update the terminal UI with the initial observation if it exists
        if terminal_ui:
            terminal_ui.update_grid(observation)

        success = False
        num_steps = 0
        while num_steps < self.max_steps:
            action: Optional[SymbolicAction | StopAction] = model.step(observation)

            actions.append(str(action))

            if isinstance(action, StopAction):
                success = example.impossible
                break
            observation = environment.step(action)
            frames.append(observation["image"])
            self.expand_observation(example.target, observation)

            num_steps += 1
            success = self.check_done(observation["inventory"], example.target)

            # Update the terminal UI with the new observation after each step
            if terminal_ui:
                terminal_ui.update_grid(observation)
                terminal_ui.update_stats(
                    {
                        "steps": num_steps,
                    }
                )

            if success:
                break

            try:
                num_actions = get_num_actions_to_goal(
                    observation,
                    impossible=example.impossible,
                )
            except AssertionError as e:
                error_msg = (
                    f"Error getting number of actions to goal for example {example.id}"
                )
                error_msg += f"\nTarget: {example.target}"
                error_msg += f"\n\n{observation['inventory']}"
                raise ValueError(error_msg)

            num_actions_to_goal.append(num_actions)
            if num_actions == MAX_PLAN_COST:
                # if the task is now impossible
                break

        return {
            "success": success,
            "num_steps": num_steps,
            "target": example.target,
            "recipe_type": example.recipe_type,
            "complexity": example.complexity_split,
            "frames": frames,
            "example_id": example.id,
            "num_actions_to_goal": num_actions_to_goal,
            "actions_history": actions,
            "cummulative_regret": cummulative_regret(num_actions_to_goal),
            "env_actions": count_env_actions(actions),
        }


#### Tools ####
class PlancraftMove(Tool):
    name = "move"
    description = "Transfer a specific quantity of an item from one slot to another"
    inputs = {
        "slot_from": {
            "type": "string",
            "description": "The slot to move the item from (A1, A2, B1, B2, etc.)",
        },
        "slot_to": {
            "type": "string",
            "description": "The slot to move the item to (A1, A2, B1, B2, etc.)",
        },
        "quantity": {
            "type": "integer",
            "description": "The number of items to move",
        },
    }
    output_type = "any"
    skip_forward_signature_validation = True

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, slot_from: str, slot_to: str, quantity: int, **kwargs) -> Any:
        return MoveAction(slot_from=slot_from, slot_to=slot_to, quantity=quantity)


class PlancraftSmelt(Tool):
    name = "smelt"
    description = "Smelt an item in a furnace and moves the output to a specific slot"
    inputs = {
        "slot_from": {
            "type": "string",
            "description": "The slot to smelt the item from (A1, A2, B1, B2, etc.)",
        },
        "slot_to": {
            "type": "string",
            "description": "The slot to smelt the item to (A1, A2, B1, B2, etc.)",
        },
        "quantity": {
            "type": "integer",
            "description": "The number of items to smelt",
        },
    }
    output_type = "any"
    skip_forward_signature_validation = True

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, slot_from: str, slot_to: str, quantity: int, **kwargs) -> Any:
        return SmeltAction(slot_from=slot_from, slot_to=slot_to, quantity=quantity)


class PlancraftImpossible(Tool):
    name = "impossible"
    description = (
        "Stop task if it is certain that it is impossible with given inventory"
    )
    inputs = {
        "reason": {
            "type": "string",
            "description": "The reason why the action is impossible",
        },
    }
    output_type = "any"
    skip_forward_signature_validation = True

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, reason: str, **kwargs) -> Any:
        return StopAction(reason=reason)


PLANCRAFT_BASE_TOOLS = {
    "move": PlancraftMove(),
    "smelt": PlancraftSmelt(),
    "impossible": PlancraftImpossible(),
}


def plancraft_planner(observation, recipe_name=None, clear_first=False) -> dict:
    """
    Use the planner to get a plan for the given observation.
    """
    if recipe_name:
        try:
            new_obs = deepcopy(observation)
            new_obs["target"] = recipe_name
            subplans, plan, items = get_subplans(
                new_obs, return_items=True, clear_first=clear_first
            )
        except:
            return {"subplans": [["impossible"]], "plan": []}

    else:
        subplans, plan, items = get_subplans(
            observation, return_items=True, clear_first=clear_first
        )

    for i in range(len(subplans)):
        for j in range(len(subplans[i])):
            subplans[i][j] = subplans[i][j].replace("]", "").replace("[", "")
    recipe = ""
    if recipe_name is not None:
        recipe = gold_search_recipe(recipe_name)
    return {
        "subplans": subplans,
        "plan": plan,
        "recipe": recipe,
        "subplan_items": items,
    }


def hardcode_plancraft_teacher_answer(teacher_mode, recipe, planner_output):
    # Hardcoded answers for specific targets
    if (
        len(planner_output["subplans"]) == 1
        and len(planner_output["subplans"][0]) == 1
        and "impossible" in planner_output["subplans"][0][0]
    ):
        return f"It is impossible to craft {recipe} with the current inventory."
    answer = f"To craft a {recipe}, follow these steps:\n"
    match teacher_mode:
        case "executable":
            step_counter = 1
            for subplan in planner_output["subplans"]:
                for action in subplan:
                    answer += f"{step_counter}. {action}\n"
                    step_counter += 1
        case "partially-executable":
            # same as executable but uses the item names instead of from location
            step_counter = 1
            for subplan, items in zip(
                planner_output["subplans"], planner_output["subplan_items"]
            ):
                for action, item in zip(subplan, items):
                    action_name = action.split(":")[0].strip()
                    destination = action.split("to")[1].split("with")[0].strip()
                    if destination[0] == "I":
                        destination = "a free inventory slot"
                    new_action = f"{action_name} {item} to {destination}"
                    answer += f"{step_counter}. {new_action}\n"
                    step_counter += 1

        case "subgoal-partially-executable":
            # separates the steps into subgoals, but still uses the item names instead of from location
            overall_counter = 1
            for subplan, recipe, items in zip(
                planner_output["subplans"],
                planner_output["plan"],
                planner_output["subplan_items"],
            ):
                if isinstance(recipe[0], BaseRecipe):
                    answer += f"{overall_counter}. Craft {recipe[0].result.item}\n"
                else:
                    answer += f"{overall_counter}. Clear crafting grid\n"
                    if isinstance(items[0], list):
                        items = [item for sublist in items for item in sublist]
                for i, (action, item) in enumerate(zip(subplan, items)):
                    action_name = action.split(":")[0].strip()
                    destination = action.split("to")[1].split("with")[0].strip()
                    if destination[0] == "I":
                        destination = "a free inventory slot"
                    new_action = f"{action_name} {item} to {destination}"
                    answer += f"{overall_counter}.{i + 1}. {new_action}\n"
                overall_counter += 1
        case "non-executable":
            # separates the steps into subgoals, but still uses the item names instead of from location
            overall_counter = 1
            destination_map = {
                "A1": "top left",
                "A2": "top middle",
                "A3": "top right",
                "B1": "middle left",
                "B2": "center",
                "B3": "middle right",
                "C1": "bottom left",
                "C2": "bottom middle",
                "C3": "bottom right",
            }
            for subplan, recipe, items in zip(
                planner_output["subplans"],
                planner_output["plan"],
                planner_output["subplan_items"],
            ):
                if isinstance(recipe[0], BaseRecipe):
                    answer += f"{overall_counter}. Craft {recipe[0].result.item}\n"
                else:
                    answer += f"{overall_counter}. Clear crafting grid\n"
                    if isinstance(items[0], list):
                        items = [item for sublist in items for item in sublist]
                for i, (action, item) in enumerate(zip(subplan, items)):
                    action_name = action.split(":")[0].strip()
                    destination = action.split("to")[1].split("with")[0].strip()
                    if destination[0] == "I":
                        destination = "a free inventory slot"
                    elif destination[:2] in destination_map:
                        destination = destination_map[destination[:2]]

                    new_action = f"{action_name} {item} to {destination}"
                    answer += f"{overall_counter}.{i + 1}. {new_action}\n"
                overall_counter += 1
        case "search":
            # do a look up in the gold search recipe and return
            return (
                gold_search_recipe(recipe.replace(" ", "_").strip().lower())
                .replace("[", "")
                .replace("]", "")
            )
        case _:
            raise ValueError(f"Invalid teacher mode: {teacher_mode}")
    return answer


def get_num_actions_to_goal(observation, impossible=False) -> int:
    """
    Use the planner to get a plan for the given observation.
    """
    if impossible:
        # if the task is impossible then always one action away ("impossible")
        return 1
    # get the plan
    plan, _ = get_subplans(observation)
    count = 0
    for subgoal in plan:
        for _ in subgoal:
            count += 1
    if count == 1:
        # check if the first and only action is "impossible" -- meaning the task has become impossible
        if "impossible" in subgoal[0]:
            return MAX_PLAN_COST
    return count


def cummulative_regret(distances) -> int:
    optimal = list(range(distances[0], -1, -1))
    # Pad if actual list is longer than optimal
    if len(distances) > len(optimal):
        optimal += [0] * (len(distances) - len(optimal))
    regret = [max(actual - opt, 0) for actual, opt in zip(distances, optimal)]
    return sum(regret)


def count_env_actions(actions):
    count = {
        "move": 0,
        "smelt": 0,
        "impossible": 0,
        "no_action": 0,
    }
    for action in actions:
        if action.startswith("move"):
            count["move"] += 1
        elif action.startswith("smelt"):
            count["smelt"] += 1
        elif action.startswith("impossible"):
            count["impossible"] += 1
        else:
            count["no_action"] += 1
    count["total_env_actions"] = sum(count.values())
    return count
