import os
from copy import deepcopy

import pandas as pd
from loguru import logger
from plancraft.environment.env import PlancraftEnvironment
from litellm import completion

from how.agents import AskAgent, TeacherAgent
from how.datasets.plancraft import (
    PlancraftToolsEvaluator,
    hardcode_plancraft_teacher_answer,
    plancraft_planner,
)


def validator(
    context,
    question,
    hardcoded_answer,
    teacher_answer,
    model_name: str = "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
) -> tuple[bool, str]:
    """
    Validate the teacher's answer against the planner output and example target.
    This function checks if the teacher's answer is consistent with the planner output
    and if it correctly addresses the example target.
    """
    # Create validation prompt
    validation_prompt = f"""You are a validator checking if a teacher's answer is consistent with a planner's output.

CONTEXT:
{context}

QUESTION:
{question}

PLANNER OUTPUT:
{hardcoded_answer}

TEACHER'S ANSWER:
{teacher_answer}

VALIDATION CRITERIA:
1. Does the teacher's answer align with the planner's suggested steps?
2. Does the teacher mention any items or actions not supported by the planner?
3. Is the teacher's answer relevant to crafting the target item?

Only approve answers that are consistent with the planner output.

First think step by step about the criteria then answer with either "VALID" or "INVALID"
"""

    api_base = (
        os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
        if "hosted_vllm" in model_name
        else None
    )

    try:
        response = completion(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a strict validator. "},
                {"role": "user", "content": validation_prompt},
            ],
            temperature=0.2,
            max_tokens=512,
            api_base=api_base,
        )

        validation_result = response.choices[0].message.content.strip()
        logger.info(f"Validation result: {validation_result}")
        if "INVALID" in validation_result.upper():
            return False, validation_result
        elif "VALID" in validation_result.upper():
            return True, validation_result
        else:
            logger.error("Validation result is unclear. Expected 'VALID' or 'INVALID'.")
            return False, validation_result

    except Exception as e:
        logger.error(f"Error during validation: {e}")
        # Default to invalid if validation fails
        return False, str(e)


def test_teacher_with_example(
    example_id: str,
    ask_agent: AskAgent,
    teacher: TeacherAgent,
    evaluator: PlancraftToolsEvaluator,
    recipe: str,
    clear_first: bool = False,
):
    # Find the example by ID
    example = None
    for ex in evaluator.examples:
        if ex.id == example_id:
            example = ex
            break

    if example is None:
        logger.info(f"Example with ID {example_id} not found!")
        return

    logger.info(f"Testing example {example_id} with target: {example.target}")
    logger.info("-" * 50)

    # Set up environment with example's inventory
    env = PlancraftEnvironment(
        inventory=deepcopy(example.slotted_inventory),
        resolution="high",
    )
    obs = env.step()
    evaluator.expand_observation(example.target, observation=obs)

    question_response = ask_agent.formulate_question(
        recipe=recipe,
        observation=obs,
    )
    question = question_response.content
    # logger.info(f"Question: {question}")

    # Get planner output for context
    planner_output = plancraft_planner(obs, recipe_name=recipe)
    # Test hardcoded answer
    logger.info("Hardcoded teacher answer:")
    hardcoded_answer = hardcode_plancraft_teacher_answer(
        teacher.teacher_mode, recipe, planner_output
    )
    logger.info(hardcoded_answer)
    logger.info("-" * 50)
    # # Test LLM teacher
    logger.info("LLM teacher answer:")
    teacher_response = teacher.answer_question(
        question=question, observation=obs, recipe=recipe
    )
    # logger.info(teacher_response.content)
    logger.info(f"Tokens used: {teacher_response.token_usage}")

    return {
        "example": example,
        "observation": obs,
        "planner_output": planner_output,
        "teacher_answer": teacher_response,
        "question": question,
        "hardcoded_answer": hardcoded_answer,
        "context": obs["message"],
    }


if __name__ == "__main__":
    evaluator = PlancraftToolsEvaluator("teacher_judge", split="val")
    examples = [e for e in evaluator.examples if not e.impossible]
    model_name = os.environ.get(
        "MODEL_NAME", "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"
    )
    teacher_modes = [
        "non-executable",
        "subgoal-partially-executable",
        "partially-executable",
        "executable",
    ]
    results = []
    for teacher_mode in teacher_modes:
        logger.info(f"Testing teacher mode: {teacher_mode}")
        logger.info(f"Processing {len(examples)} examples...")
        teacher = TeacherAgent(
            model_name=model_name, teacher_mode=teacher_mode, hardcoded_mode=False
        )
        ask_agent = AskAgent(
            model_name=model_name,
            hardcoded_mode=False,
        )
        for i, example in enumerate(examples):
            try:
                logger.info(f"Processing example {i + 1}/{len(examples)}: {example.id}")
                if example.impossible:
                    logger.info(f"Skipping impossible example {example.id}")
                    continue

                result = test_teacher_with_example(
                    example_id=example.id,
                    ask_agent=ask_agent,
                    teacher=teacher,
                    evaluator=evaluator,
                    recipe=example.target,
                )

                # Run validator
                is_valid, valid_str = validator(
                    context=result["context"],
                    question=result["question"],
                    hardcoded_answer=result["hardcoded_answer"],
                    teacher_answer=result["teacher_answer"].content,
                    model_name=model_name,
                )

                # Store results
                results.append(
                    {
                        "example_id": example.id,
                        "target": example.target,
                        "teacher_mode": teacher_mode,
                        "question": result["question"],
                        "hardcoded_answer": result["hardcoded_answer"],
                        "teacher_answer": result["teacher_answer"].content,
                        "context": result["context"],
                        "validator_result": is_valid,
                        "validation_message": valid_str,
                        "token_usage": result["teacher_answer"].token_usage,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing example {example.id}: {e}")
                # Store error result
                results.append(
                    {
                        "example_id": example.id,
                        "target": example.target,
                        "question": None,
                        "hardcoded_answer": None,
                        "teacher_answer": None,
                        "context": None,
                        "validator_result": False,
                        "validation_message": "Error during processing",
                        "token_usage": None,
                        "error": str(e),
                    }
                )

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)

    # append to CSV file
    output_file = "teachers.csv"
    df.to_csv(output_file, index=False)

    logger.info(f"Results saved to {output_file}")
    logger.info(f"Total examples processed: {len(results)}")
    logger.info(f"Valid answers: {df['validator_result'].sum()}")
    logger.info(f"Invalid answers: {(~df['validator_result']).sum()}")
