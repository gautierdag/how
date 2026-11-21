# How²: How to learn from procedural How-to questions

[![arXiv](https://img.shields.io/badge/arXiv-2510.11144-b31b1b.svg)](https://arxiv.org/abs/2510.11144)

Code for reproducing the experiments in [**How²: How to learn from procedural How-to questions**](https://arxiv.org/abs/2510.11144) (Gautier Dagan, Frank Keller, Alex Lascarides) on the [Plancraft](https://github.com/gautierdag/plancraft) environment.

## Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- `tmux` (only required if running the full suite via `experiments.sh`)
- A GPU machine to host open-weight models locally with `vllm` (or API credentials for any [LiteLLM](https://docs.litellm.ai/)-supported provider, e.g. `OPENAI_API_KEY`)
- For gated Hugging Face models such as Llama 3.3, export `HF_TOKEN` before starting vLLM:
  ```bash
  export HF_TOKEN=your_huggingface_token
  ```

## Setup

Install `uv`, `tmux`, and all Python dependencies:

```bash
bash setup.sh
```

Or, if you already have `uv` (and `tmux`) installed:

```bash
uv sync --no-install-package flash-attn
uv sync
```

## Serving a Model with vLLM

We evaluate open-weight models through a local vLLM OpenAI-compatible server on port `8000`. The commands below assume 4 GPUs; adjust `--tensor-parallel-size` to match your hardware.

**Qwen3-32B:**

```bash
uv run vllm serve Qwen/Qwen3-32B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32000 \
  --reasoning-parser qwen3
```

**Llama-3.3-70B-Instruct:**

```bash
uv run vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --chat-template chat-templates/llama32.jinja \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16000
```

By default, any model prefixed with `hosted_vllm/` connects to `http://localhost:8000/v1`. To point at a different host or port, set `VLLM_API_BASE`:

```bash
export VLLM_API_BASE="http://<host>:<port>/v1"
```

## Running the Experiments

### Full Experiment Suite

`experiments.sh` launches all baseline, teacher-type, and memory configurations in parallel as separate `tmux` windows against the running vLLM server. Start a `tmux` session first, then run:

```bash
# Default model (Llama-3.3-70B-Instruct)
bash experiments.sh

# Or pass a different model name explicitly
bash experiments.sh hosted_vllm/Qwen/Qwen3-32B
```

Stderr logs for each run are written to `logs/<run_name>.err.log`, and evaluation traces/results are saved under `output/<run_name>/` (resumable by default).

### Single Configuration

You can run any individual setting directly with `main.py` (which evaluates both the `val` and `val.repeated` splits):

```bash
# Base agent (no memory / teacher)
uv run main.py \
  --model-name hosted_vllm/meta-llama/Llama-3.3-70B-Instruct \
  --no-memory \
  --tools think

# Memory + parsing + relevance check with a partially-executable teacher
uv run main.py \
  --model-name hosted_vllm/meta-llama/Llama-3.3-70B-Instruct \
  --memory \
  --teacher-type partially-executable \
  --parse-teacher-answer \
  --relevance-check \
  --hardcoded-mode \
  --tools think
```

Run `uv run main.py --help` to see all available options (teacher types, `--skip-cache`, `--clear-first`, `--curriculum`, `--wandb-mode`, `--output-dir`, etc.).

### Teacher Validation

To evaluate LLM-generated teacher answers against the planner output using an LLM-as-a-judge validator:

```bash
uv run teacher_judge.py
# Optionally override the model:
MODEL_NAME="hosted_vllm/Qwen/Qwen3-32B" uv run teacher_judge.py
```

Results are written to `teachers.csv`.

## Repository Structure

- `main.py` — Main evaluation entrypoint and CLI
- `experiments.sh` — Launches the full matrix of experiments in `tmux`
- `teacher_judge.py` — Validates LLM teacher outputs against the planner
- `how/agents.py` — `MainAgent`, `AskAgent`, `TeacherAgent`, and `MemoryCache`
- `how/config.py` — Configuration dataclasses and enums
- `how/datasets/plancraft.py` — Plancraft evaluator, tools, and planner wrapper
- `how/datasets/data/val.repeated.json` — Repeated validation split used in the paper
- `how/prompts/plancraft/` — Prompt templates for the main, ask, and teacher agents
- `chat-templates/` — Jinja chat templates for Llama tool calling

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{dagan2026how2learnproceduralhowto,
      title={$How^{2}$: How to learn from procedural How-to questions}, 
      author={Gautier Dagan and Frank Keller and Alex Lascarides},
      year={2026},
      eprint={2510.11144},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.11144}, 
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
