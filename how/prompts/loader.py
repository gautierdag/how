from pathlib import Path
from typing import Any, Literal

import yaml
from jinja2 import StrictUndefined, Template


def load_prompts(
    agent_type: Literal["main", "ask", "teacher"] = "main",
    environment="plancraft",
) -> dict[str, str]:
    # Get the prompts directory
    file_path = Path(__file__).parent / environment / f"{agent_type}.yaml"
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    with open(file_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data


def populate_template(template: str, variables: dict[str, Any]) -> str:
    """
    Format a prompt template with Jinja-style placeholders.
    """
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(
            f"Error during jinja template rendering: {type(e).__name__}: {e}"
        )
