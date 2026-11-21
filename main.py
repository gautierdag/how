import warnings
from dataclasses import asdict
from typing import Optional

import typer
from loguru import logger
from smolagents import Tool

import wandb
from how.agents import (
    AskAgent,
    MainAgent,
    MemoryCache,
    ReadMemoryTool,
    TeacherAgent,
    ThinkTool,
)
from how.config import (
    ComplexityFilter,
    Config,
    TeacherType,
    WandbMode,
    create_config_from_args,
    generate_run_name,
    store_config,
)
from how.datasets.plancraft import PlancraftToolsEvaluator
from how.utils.stats import calculate_evaluation_stats
from how.utils.terminal_ui import TerminalUI, TerminalUISink

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:.*",
    category=UserWarning,
    module="pydantic.main",
)


def get_tools(config: Config) -> tuple[list[Tool], Optional[MemoryCache]]:
    """Get the tools for the agent based on the configuration."""
    tools = []
    memory_cache = None
    logger.info(f"Using tools: {config.tools}")
    if config.memory:
        teacher_agent = TeacherAgent(
            model_name=config.model_name,
            teacher_mode=config.teacher_type.value,
            hardcoded_mode=config.hardcoded_mode,
            clear_first=config.clear_first,
        )
        ask_agent = AskAgent(
            model_name=config.model_name,
            hardcoded_mode=config.hardcoded_mode,
        )
        memory_cache = MemoryCache(
            teacher_agent=teacher_agent,
            ask_agent=ask_agent,
            parse_teacher_answer=config.parse_teacher_answer,
            skip_cache=config.skip_cache,
            relevance_check=config.relevance_check,
        )
        read_mem_tool = ReadMemoryTool(memory_cache=memory_cache)
        tools.append(read_mem_tool)
    if "think" in config.tools:
        think_tool = ThinkTool()
        tools.append(think_tool)
    return tools, memory_cache


def evaluate(
    config: Config,
):
    """Run evaluation based on the provided config and model."""
    run_name = generate_run_name(config)

    logger.info(f"Starting evaluation run: {run_name}")

    wandb_run = wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=asdict(config),
        tags=[
            config.teacher_type,
            config.model_name.split("/")[-1],
        ],
        mode=config.wandb_mode,
    )

    # Initialize the terminal UI
    terminal_ui = TerminalUI(title=f"Plancraft - {run_name}")

    # Configure loguru to use our terminal UI sink
    terminal_ui_sink = TerminalUISink(terminal_ui)
    logger.remove()
    logger.add(terminal_ui_sink, level="INFO")
    logger.add(lambda msg: print(msg), level="WARNING")

    evaluator = PlancraftToolsEvaluator(
        output_dir=config.output_dir,
        run_name=run_name,
        max_steps=30,
        use_text_inventory=True,
        resume=config.resume,
        split=config.split,
        curriculum=config.curriculum,
    )
    tools, memory_cache = get_tools(config)
    agent = MainAgent(
        model_name=config.model_name, tools=tools, ask_first=config.ask_first
    )

    # Store configuration
    logger.info(f"Storing configuration to {evaluator.output_dir}/config.json")
    store_config(config, f"{evaluator.output_dir}/config.json")

    examples = evaluator.examples

    # Apply complexity filter if specified
    if config.complexity_filter.value != "none":
        examples = [
            e for e in examples if e.complexity_split == config.complexity_filter.value
        ]
        logger.info(
            f"Filtered to {len(examples)} {config.complexity_filter.value} examples"
        )
    else:
        logger.info(f"Using all {len(examples)} examples")

    # Update initial stats
    terminal_ui.update_stats(
        {
            "total_examples": len(examples),
            "examples_completed": 0,
            "success_rate": 0.0,
            "steps": 0,
        }
    )

    # Start the terminal UI
    with terminal_ui:
        # Store all results for final statistics calculation
        all_results = []
        for generation in range(config.num_generations):
            # clear memory cache
            if memory_cache:
                memory_file = f"{evaluator.output_dir}/{generation}/memory.json"
                memory_cache.reset()
                if config.resume:
                    memory_cache.load_from_file(memory_file)

            evaluator.generation_number = generation
            successful_examples = 0
            logger.info(f"Starting generation for {len(examples)} examples")
            # Get examples that haven't been processed
            for i, example in enumerate(examples):
                # Update current example in UI
                terminal_ui.update_stats(
                    {
                        "current_example": f"{example.id} (Target: {example.target})",
                        "is_impossible": example.impossible,
                        "examples_completed": i,
                        "steps": 0,
                    }
                )

                # Check if the example has already been processed
                if res := evaluator.load_results_dict(example):
                    # Add example id and complexity for stats calculation
                    res["order_id"] = i
                    all_results.append(res)
                    if res.get("success", False):
                        successful_examples += 1
                    continue

                # clear agent history
                agent.reset()

                # Process the example
                result = evaluator.eval_example(example, agent, terminal_ui=terminal_ui)

                # Update the success count
                if result.get("success", False):
                    successful_examples += 1

                # Update trace information with both agent and memory traces
                detailed_trace = {
                    "main_agent": agent.trace(),
                }
                if config.memory:
                    detailed_trace["memory_cache"] = memory_cache.trace()
                    memory_cache.reset_interactions()
                    terminal_ui.increment_stats(
                        "memory_calls",
                        detailed_trace["memory_cache"]["memory_stats"][
                            "total_memory_calls"
                        ],
                    )
                    terminal_ui.increment_stats(
                        "cache_hits",
                        detailed_trace["memory_cache"]["memory_stats"]["cache_hits"],
                    )
                    terminal_ui.increment_stats(
                        "cache_misses",
                        detailed_trace["memory_cache"]["memory_stats"]["cache_misses"],
                    )
                    terminal_ui.increment_stats(
                        "tokens_used_memory",
                        detailed_trace["memory_cache"]["memory_stats"]["total_tokens"],
                    )
                    terminal_ui.increment_stats(
                        "tokens_used",
                        detailed_trace["memory_cache"]["memory_stats"]["total_tokens"],
                    )

                # Save the combined trace to the results file
                result["trace"] = detailed_trace
                result["order_id"] = i
                # save images to file
                if config.save_images:
                    evaluator.save_images(example, result["frames"])
                if "frames" in result:
                    del result["frames"]  # can't save images to json
                evaluator.save_results_dict(example, result)

                # Add example id and complexity for stats calculation
                all_results.append(result)

                # Update success rate in UI
                terminal_ui.update_stats(
                    {
                        "success_rate": successful_examples / (i + 1),
                    }
                )

                terminal_ui.increment_stats(
                    "tokens_used_agent", result["trace"]["main_agent"]["tokens_used"]
                )
                terminal_ui.increment_stats(
                    "tokens_used", result["trace"]["main_agent"]["tokens_used"]
                )

                # update memory cache
                if memory_cache:
                    memory_cache.save_to_file(memory_file)

    # Calculate final statistics using the new function
    num_successes = sum(1 for result in all_results if result.get("success", False))
    stats = {
        "success_rate": num_successes / len(all_results) if all_results else 0.0,
    }
    try:
        calculate_evaluation_stats(all_results)
    except Exception as e:
        logger.error(f"Error calculating evaluation stats: {e}")

    wandb_run.log(stats)
    wandb_run.finish()
    return stats


app = typer.Typer()


@app.command()
def run(
    # General configuration
    environment: str = typer.Option("plancraft", help="Environment to evaluate on"),
    output_dir: str = typer.Option("output", help="Directory to save evaluation outputs"),
    num_generations: int = typer.Option(3, help="Number of generations to run"),
    save_images: bool = typer.Option(True, help="Save images of environment states"),
    curriculum: bool = typer.Option(False, help="Whether to use curriculum learning"),
    complexity_filter: ComplexityFilter = typer.Option(
        ComplexityFilter.NONE, help="Filter examples by complexity"
    ),
    resume: bool = typer.Option(True, help="Resume from previous run"),
    # Model configuration
    tools: list[str] = typer.Option([], help="Tools to use (e.g., think, search)"),
    memory: bool = typer.Option(False, help="Enable memory"),
    parse_teacher_answer: bool = typer.Option(
        False, help="Parse teacher answer for better understanding"
    ),
    teacher_type: TeacherType = typer.Option(
        TeacherType.EXECUTABLE, help="Type of teacher agent"
    ),
    model_name: str = typer.Option(
        "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct", help="Model to use"
    ),
    hardcoded_mode: bool = typer.Option(
        False, help="Use hardcoded mode for the Ask/Teacher interactions"
    ),
    skip_cache: bool = typer.Option(
        False, help="Skip the memory cache for the evaluation. Just Ask baseline."
    ),
    relevance_check: bool = typer.Option(
        False, help="Check relevance of the teacher's answer to the task"
    ),
    clear_first: bool = typer.Option(
        False, help="Whether to tell agent to clear the inventory first before planning"
    ),
    ask_first: bool = typer.Option(
        False, help="Fixed policy for agent to always ask first"
    ),
    # Logging configuration
    wandb_mode: WandbMode = typer.Option(WandbMode.DISABLED, help="Wandb mode"),
    wandb_project: str = typer.Option("how", help="Wandb project name"),
    # Run name
    run_name: Optional[str] = typer.Option(None, help="Custom run name"),
):
    """Run the evaluation with the given configuration."""
    # Create config from args
    for split in ["val", "val.repeated"]:
        config = create_config_from_args(
            environment=environment,
            output_dir=output_dir,
            num_generations=num_generations,
            save_images=save_images,
            split=split,
            curriculum=curriculum,
            complexity_filter=complexity_filter,
            resume=resume,
            tools=tools,
            memory=memory,
            parse_teacher_answer=parse_teacher_answer,
            teacher_type=teacher_type,
            model_name=model_name,
            wandb_mode=wandb_mode,
            wandb_project=wandb_project,
            run_name=run_name,
            hardcoded_mode=hardcoded_mode,
            skip_cache=skip_cache,
            relevance_check=relevance_check,
            clear_first=clear_first,
            ask_first=ask_first,
        )
        # Run evaluation
        results = evaluate(config)
        logger.info(f"Evaluation complete. Success rate: {results['success_rate']:.2%}")


if __name__ == "__main__":
    app()
