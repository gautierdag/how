import json
import os
import re
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass

from litellm import batch_completion, completion
from litellm.exceptions import BadRequestError, ContextWindowExceededError
from litellm.types.utils import Message
from loguru import logger
from smolagents import Tool
from smolagents.models import get_tool_json_schema
from how.datasets.plancraft import get_objective_str
from collections import Counter
from how.datasets.plancraft import (
    PLANCRAFT_BASE_TOOLS,
    hardcode_plancraft_teacher_answer,
    plancraft_planner,
)
from how.prompts.loader import load_prompts, populate_template

SUPPORTED_ENVIRONMENTS = ["plancraft"]


class ThinkTool(Tool):
    name = "think"
    description = "Generate thoughts to help you decide on the next action"
    inputs = {
        "thought": {
            "type": "string",
            "description": "<thought message>",
        },
    }
    output_type = "string"
    skip_forward_signature_validation = True

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, thought: str, **kwargs) -> str:
        return thought


@dataclass
class ToolResponse:
    content: str
    error: bool = False
    token_usage: int = 0


@dataclass
class StructuredMemoryContent(ToolResponse):
    related_items: list[str] = None


def serialise_messages(obj):
    if isinstance(obj, Message):
        return obj.json()
    elif isinstance(obj, list):
        return [serialise_messages(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialise_messages(v) for k, v in obj.items()}
    return obj


def count_dialogue_steps(dialogue_history):
    count = {
        "total_messages": len(dialogue_history),
        "tool_calls": 0,
        "tool_failures": 0,
    }
    for message in dialogue_history:
        if message["role"] == "user":
            count["user_steps"] = count.get("user_steps", 0) + 1
        elif message["role"] == "assistant":
            count["assistant_steps"] = count.get("assistant_steps", 0) + 1
            if (
                "tool_calls" in message
                and message["tool_calls"]
                and len(message["tool_calls"]) > 0
            ):
                count["tool_calls"] += len(message["tool_calls"])
                tool_name = message["tool_calls"][0]["function"]["name"]
                count[f"{tool_name}_calls"] = count.get(f"{tool_name}_calls", 0) + 1
        elif message["role"] == "tool":
            if "error" in message["content"].lower():
                count["tool_failures"] += 1
    return count


class TeacherAgent:
    def __init__(
        self,
        model_name: str,
        teacher_mode: str = "executable",
        environment: str = "plancraft",
        hardcoded_mode: bool = False,
        clear_first: bool = False,
    ):
        self.model_name = model_name
        self.hardcoded_mode = hardcoded_mode
        self.environment = environment
        self.clear_first = clear_first
        self.prompts = load_prompts(
            agent_type="teacher",
            environment=environment,
        )
        assert teacher_mode in self.prompts, f"Invalid teacher_mode: {teacher_mode}"
        if environment == "plancraft":
            self.planner_func = plancraft_planner
        self.teacher_mode = teacher_mode
        self.api_base = (
            os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
            if "hosted_vllm" in self.model_name
            else None
        )

    def clean_inventory_message(self, observation) -> str:
        """
        Generate a message summarizing the inventory
        Without item locations, just item types and quantities.
        """
        inventory_counter_message = (
            f"{get_objective_str(observation['target'])}\ninventory:"
        )
        type_counts = Counter()
        for item in observation["inventory"].values():
            type_counts[item["type"]] += item["quantity"]
        for item_type, count in type_counts.items():
            inventory_counter_message += f"\n - {count} {item_type}"
        return inventory_counter_message.strip()

    def answer_question(
        self,
        question: str,
        observation: dict,
        recipe: str,
        **kwargs,
    ) -> ToolResponse:
        # Get context and plan
        try:
            planner_output = self.planner_func(
                observation, recipe_name=recipe, clear_first=self.clear_first
            )
        except:
            return ToolResponse(
                content=f"It is impossible to craft {recipe} with the current inventory."
            )

        if not planner_output["plan"]:
            response_text = (
                f"It is impossible to craft {recipe} with the current inventory."
            )
            return ToolResponse(content=response_text)

        if self.hardcoded_mode:
            # If hardcoded mode is enabled, use template with planner to answer the question
            assert (
                self.environment == "plancraft"
            ), "Hardcoded mode only supported for plancraft"
            # Use hardcoded answers for the teacher agent
            answer = hardcode_plancraft_teacher_answer(
                teacher_mode=self.teacher_mode,
                recipe=recipe,
                planner_output=planner_output,
            )
            return ToolResponse(
                content=answer,
                token_usage=0,
            )

        # use the hardcoded answer to generate the planner string given to Teacher LLM
        planner_str = hardcode_plancraft_teacher_answer(
            teacher_mode=self.teacher_mode,
            recipe=recipe,
            planner_output=planner_output,
        )
        # Load the prompt based on teacher_mode
        prompt_template = self.prompts[self.teacher_mode]

        # clear message of item locations to prevent leakage of item locations
        if self.teacher_mode != "executable":
            context = self.clean_inventory_message(observation)
        else:
            context = observation["message"]

        system_prompt = populate_template(
            prompt_template,
            variables={
                "context": context,
                "planner_str": planner_str,
            },
        )
        # Generate response using the appropriate teacher mode
        response = completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            # max_tokens=512,
            api_base=self.api_base,
            stop=["Note:"],
        )
        response_text = response.choices[0].message.content.strip("\n")
        logger.info(f"Teacher response: {response_text}")

        return ToolResponse(
            content=response_text, token_usage=response.usage.total_tokens
        )


class AskAgent:
    def __init__(
        self,
        model_name: str,
        environment: str = "plancraft",
        hardcoded_mode: bool = False,
    ):
        self.model_name = model_name
        self.hardcoded_mode = hardcoded_mode
        self.prompts = load_prompts(agent_type="ask", environment=environment)
        self.system_prompt = populate_template(
            self.prompts["system"],
            variables={},
        )
        self.api_base = (
            os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
            if "hosted_vllm" in self.model_name
            else None
        )

    def formulate_question(self, recipe: str, observation: dict) -> ToolResponse:
        """Generate a well-formed question about the recipe"""

        if self.hardcoded_mode:
            question = f"How do I craft {recipe}?"
            return ToolResponse(
                content=question,
                token_usage=0,  # Hardcoded questions don't use tokens
            )

        # Use the ask-questions prompt to generate a well-formed question
        ask_prompt = populate_template(
            self.prompts["ask-questions"],
            variables={"context": observation["message"], "recipe_name": recipe},
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": ask_prompt},
        ]

        response = completion(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            # max_tokens=1024,
            api_base=self.api_base,
        )
        response_text = response.choices[0].message.content
        logger.info(f"Question: {response_text}")
        return ToolResponse(
            content=response_text, token_usage=response.usage.total_tokens
        )

    def relevance_check(
        self,
        recipe: str,
        observation: dict,
        memories: list[str],
    ) -> list[ToolResponse]:
        """use the model to check if the memory is relevant to the observation"""
        relevance_check_prompt = self.prompts["relevance-check"]

        batch = []
        for memory in memories:
            prompt = populate_template(
                relevance_check_prompt,
                variables={
                    "recipe_name": recipe,
                    "context": observation["message"],
                    "memory": memory,
                },
            )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            batch.append(messages)

        # Use batch completion to check relevance of all memories in parallel
        responses = batch_completion(
            model=self.model_name,
            messages=batch,
            temperature=0.2,
            # max_tokens=128,
            api_base=self.api_base,
        )
        tool_responses = []
        for response in responses:
            response_text = response.choices[0].message.content
            tool_responses.append(
                ToolResponse(
                    content=response_text,
                    token_usage=response.usage.total_tokens,
                )
            )
        logger.info(f"Relevance check responses: {[r.content for r in tool_responses]}")
        return tool_responses

    def parse_answer(
        self, recipe: str, observation: dict, question: str, answer: str
    ) -> StructuredMemoryContent:
        """Parse the answer from the teacher agent to insert into memory"""
        # Use the write-to-memory prompt to format the answer
        write_prompt_template = self.prompts["write-to-memory"]
        write_prompt = populate_template(
            write_prompt_template,
            variables={
                "recipe_name": recipe,
                "question": question,
                "answer": answer,
                "context": observation["message"],
            },
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": write_prompt},
        ]
        response = completion(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            # max_tokens=1024,
            api_base=self.api_base,
        )
        parsed_answer = response.choices[0].message.content
        logger.info(parsed_answer)

        try:
            related_items_match = re.search(
                r"RELATED ITEMS:\s*\[([^\]]+)\]", parsed_answer
            )
            related_items = (
                [
                    item.strip().strip("'\"").lower()
                    for item in related_items_match.group(1).split(",")
                ]
                if related_items_match
                else []
            )
        except Exception as e:
            logger.warning(f"Failed to parse related items from answer: {e}")
            related_items = []

        # Ensure the recipe is included in related items
        related_items += [recipe.lower()]
        related_items = list(set(related_items))
        return StructuredMemoryContent(
            content=parsed_answer,
            related_items=related_items,
            token_usage=response.usage.total_tokens,
        )


class MemoryCache:
    # Hashmap data structure to store memory with recipe name as key
    def __init__(
        self,
        teacher_agent: TeacherAgent,
        ask_agent: AskAgent,
        parse_teacher_answer=False,
        skip_cache=False,
        relevance_check=False,
    ):
        self.teacher_agent = teacher_agent
        self.ask_agent = ask_agent
        # whether to parse the teacher's answer before committing to memory
        self.parse_teacher_answer = parse_teacher_answer
        # whether to check the relevance of the memory to decide if it should be used
        self.relevance_check = relevance_check
        # whether to skip the cache and always ask the teacher
        self.skip_cache = skip_cache
        self.memory = {}
        self.tag_to_hash = {}
        self.interactions = []
        self.stats = {
            "total_memory_calls": 0,
            "total_tokens": 0,
            "cache_misses": 0,
            "cache_hits": 0,
        }
        self.counter = 0
        self.teacher_interactions = 0

    def commit_to_memory(self, message: str, tags: list[str]) -> str:
        """Commit a message to memory under the given tags"""
        hash_id = str(uuid.uuid4())
        self.memory[hash_id] = {
            "content": message,
            "created_at": self.counter,
        }
        for tag in tags:
            tag = tag.lower().strip()
            if tag not in self.tag_to_hash:
                self.tag_to_hash[tag] = [hash_id]
            else:
                self.tag_to_hash[tag].append(hash_id)

        self.counter += 1
        return f"Added to memory: {message}..."

    def search_memory(self, tag: str) -> list[str]:
        """Search memory with a tag and returns list of matches"""
        found_content = []
        if tag in self.tag_to_hash:
            for hash_id in self.tag_to_hash[tag]:
                if hash_id in self.memory:
                    found_content.append(self.memory[hash_id]["content"])
        return found_content

    def cache_miss(self, recipe: str, observation: dict) -> ToolResponse:
        # Track this interaction
        interaction = {"tag": recipe, "timestamp": self.counter, "type": "cache_miss"}

        # Ask the teacher for information
        question = self.ask_agent.formulate_question(
            recipe,
            observation=observation,
        )
        interaction["question"] = question.content

        answer = self.teacher_agent.answer_question(
            question.content,
            observation=observation,
            recipe=recipe,
        )
        interaction["answer"] = answer.content
        self.teacher_interactions += 1

        # Update stats
        self.stats["total_memory_calls"] += 1
        self.stats["total_tokens"] += question.token_usage + answer.token_usage
        self.stats["cache_misses"] += 1
        interaction["tokens_used"] = {
            "question": question.token_usage,
            "answer": answer.token_usage,
        }

        if "impossible" in answer.content:
            # don't save negative to memory/parse
            # TODO: could experiment with counter factuals
            self.interactions.append(interaction)
            return ToolResponse(content=answer.content)

        # Parse the answer
        if self.parse_teacher_answer:
            parsed_answer = self.ask_agent.parse_answer(
                recipe=recipe,
                observation=observation,
                question=question.content,
                answer=answer.content,
            )
            interaction["parsed_answer"] = parsed_answer.content
            interaction["tokens_used"]["parse"] = parsed_answer.token_usage
            if self.relevance_check:
                self.commit_to_memory(
                    parsed_answer.content, tags=parsed_answer.related_items
                )
            else:
                # if no relevance check, just use the recipe as tag,
                # otherwise irrelevant memory can't be filtered out
                self.commit_to_memory(parsed_answer.content, tags=[recipe])
            memory = parsed_answer.content
        else:
            self.commit_to_memory(answer.content, tags=[recipe])
            memory = answer.content

        self.interactions.append(interaction)

        return ToolResponse(
            content=memory,
        )

    def retrieve_memory(
        self,
        recipe: str,
        observation: dict = None,
    ) -> ToolResponse:
        # If memory is empty or recipe not found in memory or skip_cache is True
        memories = self.search_memory(recipe)
        if self.skip_cache or not memories:
            return self.cache_miss(recipe, observation)

        if self.relevance_check:
            # If relevance check is enabled, check each memory if they are relevant to the observation
            relevance_checks = self.ask_agent.relevance_check(
                recipe, observation=observation, memories=memories
            )
            relevant_memories = []
            for relevance_answer in relevance_checks:
                self.stats["total_tokens"] += relevance_answer.token_usage
                if "yes" in relevance_answer.content.lower():
                    # If the memory is relevant, add it context
                    relevant_memories.append(relevance_answer.content)

            # If no memory is relevant, ask the teacher
            if not relevant_memories:
                return self.cache_miss(recipe, observation)

        memory_content = "\n\n".join(memories)

        self.stats["cache_hits"] += 1
        self.stats["total_memory_calls"] += 1

        logger.info(f"Cache hit for recipe: {recipe}")
        logger.info(memory_content)

        # If tag found, return the memory entry
        return ToolResponse(content=memory_content)

    def save_to_file(self, file_path: str) -> None:
        logger.debug(f"Saving memory to {file_path}")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(
                {
                    "memory": self.memory,
                    "counter": self.counter,
                    "tag_to_hash": self.tag_to_hash,
                },
                f,
                indent=2,
            )

    def load_from_file(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            logger.warning(f"Memory file {file_path} does not exist")
            return
        logger.debug(f"Loading memory from {file_path}")
        # Load memory
        with open(file_path, "r") as f:
            data = json.load(f)
            self.memory = data.get("memory", {})
            self.counter = data.get("counter", 0)
            self.tag_to_hash = data.get("tag_to_hash", {})
        logger.debug(f"Memory loaded with {len(self.memory)} entries")

    def reset(self) -> None:
        logger.info("Resetting memory to empty")
        self.memory = {}
        self.tag_to_hash = {}
        self.counter = 0
        self.reset_interactions()
        logger.info("Memory reset")

    def reset_interactions(self) -> None:
        logger.info("Resetting trace interactions")
        self.interactions = []
        self.teacher_interactions = 0
        self.stats = {
            "total_memory_calls": 0,
            "total_tokens": 0,
            "cache_misses": 0,
            "cache_hits": 0,
        }
        logger.info("Interactions reset")

    def trace(self) -> dict:
        self.stats["average_tokens_per_interaction"] = (
            self.stats["total_tokens"] / self.teacher_interactions
            if self.teacher_interactions > 0
            else 0
        )
        return {
            "memory_size": len(self.memory),
            "interactions": self.interactions,
            "memory_stats": self.stats,
        }


class ReadMemoryTool(Tool):
    name = "read_memory"
    description = "Search a database to retrieve memories/instructions for a given recipe. Call this first or if unsure."
    inputs = {
        "recipe": {
            "type": "string",
            "description": "Recipe name to search in memory.",
        },
    }
    output_type = "string"
    skip_forward_signature_validation = True

    def __init__(self, memory_cache: MemoryCache):
        super().__init__()
        self.memory_cache = memory_cache

    def __call__(self, recipe: str, observation: dict) -> str:
        recipe_name = recipe.replace(" ", "_").lower().strip()
        return self.memory_cache.retrieve_memory(recipe_name, observation)


class MainAgent:
    def __init__(
        self,
        model_name: str,
        tools: list = [],
        environment: str = "plancraft",
        max_messages_window: int = 30,
        max_non_env_calls: int = 2,
        ask_first: bool = False,
    ):
        self.model_name = model_name
        self.environment = environment
        self.tools = {tool.name: tool for tool in tools}
        self.trajectory_solvable = False
        self.ask_first = ask_first

        if self.environment == "plancraft":
            self.tools |= PLANCRAFT_BASE_TOOLS
            self.environment_actions = list(PLANCRAFT_BASE_TOOLS.keys())
        else:
            raise ValueError(
                f"Invalid environment: {self.environment}. Supported environments: plancraft"
            )

        self.history = []
        self.tokens_used = 0
        self.prompts = load_prompts(agent_type="main", environment=environment)
        self.system_prompt = populate_template(
            self.prompts["system"],
            variables={"tools": self.tools},
        )
        self.max_messages_window = max_messages_window
        self.max_non_env_calls = max_non_env_calls
        self.api_base = (
            os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
            if "hosted_vllm" in self.model_name
            else None
        )
        self.asked_teacher = False

    def reset(self):
        self.history = []
        self.tokens_used = 0
        # false -- assumes the task **could be** unsolvable
        self.trajectory_solvable = False
        self.asked_teacher = False

    def trace(self):
        dialogue_history = serialise_messages(deepcopy(self.history))
        return {
            "history": dialogue_history,
            "tokens_used": self.tokens_used,
            "dialogue_stats": count_dialogue_steps(dialogue_history),
        }

    def get_message_window(
        self,
        window_size: int = None,
    ) -> list[dict]:
        effective_window_size = window_size or self.max_messages_window
        message_window = self.history[-effective_window_size:]
        # ensure the first message is user
        if len(message_window) > 0:
            for i, message in enumerate(message_window):
                if message["role"] == "user":
                    break
            message_window = message_window[i:]
        # add the system prompt
        system_message = {"role": "system", "content": self.system_prompt}
        return [system_message] + message_window

    def safe_completion(self, **kwargs):
        """Wrapper for completion that handles context window exceeded errors"""
        errors = []
        for attempt in range(5):  # Max 5 attempts
            try:
                return completion(**kwargs)
            except ContextWindowExceededError:
                errors.append("ContextWindowExceededError")
                # Try with progressively smaller message windows
                logger.warning(f"Context window exceeded, attempt {attempt + 1}/5")
                # If this is our completion call, reduce the message window
                if "messages" in kwargs and len(kwargs["messages"]) > 2:
                    new_window_size = max(
                        2, self.max_messages_window - ((attempt + 1) * 2)
                    )
                    truncated_messages = self.get_message_window(
                        window_size=new_window_size
                    )
                    kwargs["messages"] = truncated_messages
                else:
                    # Can't reduce further
                    raise

            except BadRequestError:
                # Handle bad request errors, likely due to invalid tool calls or arguments
                logger.error(
                    f"Bad request error on attempt {attempt + 1}/5. Check tool calls and arguments."
                )
                time.sleep(1)  # Wait before retrying
                errors.append("BadRequestError")

        # If all attempts failed, raise the last error
        raise ValueError(
            f"Failed to complete after 5 attempts. Errors: {errors}. Last attempt: {kwargs.get('messages', [])}"
        )

    def get_tools(self) -> list[Tool]:
        # If ask_first is enabled and we haven't asked the teacher yet, limit tool options to force asking
        if self.ask_first and not self.asked_teacher:
            return [
                tool
                for tool in self.tools.values()
                if tool.name == "read_memory" or tool.name == "ask"
            ]
        # If trajectory is solvable, filter out the "impossible" tool
        if self.trajectory_solvable:
            return [tool for tool in self.tools.values() if tool.name != "impossible"]
        return list(self.tools.values())

    def execute_tool(self, tool_call, **kwargs) -> ToolResponse:
        function_name = tool_call.function.name
        # check if the function is in the tools
        if function_name not in self.tools:
            return ToolResponse(
                content=f"Error: {function_name} not available. Available actions: {self.tools.keys()}",
                error=True,
            )
        function_to_call = self.tools[function_name]

        function_kwargs = json.loads(tool_call.function.arguments)

        # we are asking the teacher
        if function_name in ["ask", "read_memory"] and not self.asked_teacher:
            self.asked_teacher = True

        try:
            tool_response = function_to_call(**function_kwargs, **kwargs)
            if isinstance(tool_response, ToolResponse):
                return tool_response
            # otherwise, assume the function returns a string
            return ToolResponse(content=tool_response)
        except Exception as e:
            logger.debug(
                f"Error calling function {function_name} with args {function_kwargs}: {e}"
            )
            if function_name in ["read_memory", "ask"]:
                raise e
            return ToolResponse(
                content=f"Error calling function {function_name} with args {str(function_kwargs)}: {str(e)}",
                error=True,
            )

    def step(
        self,
        observation: dict,
    ):
        i = 0
        self.history.append({"role": "user", "content": observation["message"]})
        while i < self.max_non_env_calls:
            messages = self.get_message_window()
            response = self.safe_completion(
                model=self.model_name,
                messages=messages,
                temperature=0.6,
                # max_tokens=2048,
                stop=["Final Answer", "```", "<result>"],
                top_p=1,
                tools=[get_tool_json_schema(t) for t in self.get_tools()],
                # tool_choice="required",
                api_base=self.api_base,
            )
            self.tokens_used += response.usage.total_tokens
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            if tool_calls and len(tool_calls) > 1:
                # NOTE: if there are multiple tool calls, we only keep the first one
                response_message.tool_calls = [tool_calls[0]]
            elif not tool_calls:
                # ensure model returns a tool call
                logger.info("Model did not return a tool call")

            # logging
            out_str = ""
            if response_message.content:
                out_str += response_message.content
            if tool_calls:
                out_str += f"\n{tool_calls[0].function.name}({tool_calls[0].function.arguments})"

            if response_message.get("reasoning_content", None):
                logger.info(f"Reasoning: {response_message.reasoning_content}")
            logger.info(out_str)
            self.history.append(response_message)

            # process the tool/function call
            if tool_calls:
                tool_call = tool_calls[0]
                tool_response = self.execute_tool(tool_call, observation=observation)

                if (
                    not tool_response.error
                    and tool_call.function.name in self.environment_actions
                ):
                    self.history.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": str(tool_response.content),
                        }
                    )
                    # return if the tool call is an environment action
                    return tool_response.content
                # if the tool call is not an environment action, we continue
                tool_call_response = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": tool_response.content,
                }
                self.history.append(tool_call_response)
            else:
                # if no tool call, assume the model is thinking
                self.history.append(
                    {
                        "role": "user",
                        "content": "No tool call found. Use function calling using the JSON format to call a tool.",
                    }
                )
            i += 1
        return None
