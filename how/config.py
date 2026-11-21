import json
import os
from typing import Optional
from dataclasses import asdict, dataclass, field
from enum import Enum


class TeacherType(str, Enum):
    EXECUTABLE = "executable"
    PARTIALLY_EXECUTABLE = "partially-executable"
    SUBGOAL_PARTIALLY_EXECUTABLE = "subgoal-partially-executable"
    NON_EXECUTABLE = "non-executable"
    SEARCH = "search"
    NONE = "none"


class ComplexityFilter(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    NONE = "none"


class WandbMode(str, Enum):
    DISABLED = "disabled"
    ONLINE = "online"
    OFFLINE = "offline"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    VAL_SMALL = "val.small"
    TEST_SMALL = "test.small"
    VAL_REPEATED = "val.repeated"


@dataclass
class Config:
    # General configuration
    environment: str = "plancraft"
    output_dir: str = "output"
    num_generations: int = 3
    save_images: bool = False
    split: Split = Split.VAL
    curriculum: bool = False
    complexity_filter: ComplexityFilter = ComplexityFilter.NONE
    resume: bool = True

    # Model configuration
    tools: list[str] = field(default_factory=lambda: ["think"])
    memory: bool = False
    parse_teacher_answer: bool = False
    teacher_type: TeacherType = TeacherType.EXECUTABLE
    hardcoded_mode: bool = False  # templated-based teacher
    skip_cache: bool = False
    relevance_check: bool = True
    # whether to tell agent to clear the inventory first before planning
    clear_first: bool = False
    ask_first: bool = False  # fixed policy for agent to always ask first

    model_name: str = "openai/gpt-4.1-mini"

    wandb_mode: WandbMode = WandbMode.DISABLED
    wandb_project: str = "how"
    run_name: Optional[str] = None


def create_config_from_args(**kwargs) -> Config:
    """Create a Config object from parsed arguments."""
    # Filter out None values to use defaults
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return Config(**filtered_kwargs)


def enum_asdict(obj):
    def convert(val):
        if isinstance(val, Enum):
            return val.value
        if isinstance(val, list):
            return [convert(v) for v in val]
        return val

    return {k: convert(v) for k, v in asdict(obj).items()}


def store_config(config: Config, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(enum_asdict(config), f, indent=4)


def generate_run_name(config: Config) -> str:
    """Generate a run name based on the configuration (if no run name is provided)."""

    if config.run_name:
        return config.run_name

    model_name = config.model_name.split("/")[-1]

    run_name_parts = [config.teacher_type]
    if config.memory:
        run_name_parts.append("memory")
    if len(config.tools) > 0:
        run_name_parts += sorted(config.tools)
    if config.curriculum:
        run_name_parts.append("curriculum")
    if not config.parse_teacher_answer:
        run_name_parts.append("no-parse")
    if config.hardcoded_mode:
        run_name_parts.append("hardcoded")
    if config.skip_cache:
        run_name_parts.append("no-cache")
    if config.relevance_check:
        run_name_parts.append("relevance-check")
    run_name_parts.append(model_name)
    return "-".join(run_name_parts)
