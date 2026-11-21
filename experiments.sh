#!/bin/bash

# Default model name
default_model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"
# default_model="hosted_vllm/Qwen/Qwen3-32B"

# Parse command line arguments
model_name="${1:-$default_model}"

session_name=$(tmux display-message -p '#S')  # current session name

if [ -z "$session_name" ]; then
  echo "Not in a tmux session."
  exit 1
fi

# Create a logs directory if it doesn't exist
mkdir -p logs

# Function to create tmux window with experiment
create_experiment_window() {
  local run_name="$1"
  local additional_args="$2"
  
  if tmux list-windows -t "$session_name" -F '#W' | grep -qxF "${run_name}"; then
    echo "Window '${run_name}' already exists"
    return
  fi

  echo "Creating tmux window: $run_name"
  local log_file="logs/${run_name}.err.log" # Define the log
  # Redirect stderr to the specified log file, appending to it
  tmux new-window -a -t "$session_name" -n "${run_name}" \
  "uv run main.py --model-name $model_name --wandb-mode disabled --run-name $run_name --tools 'think' --num-generations 3 $additional_args 2>> \"$log_file\""
}

# Base experiment
create_experiment_window "base" "--no-memory"

for teacher_type in executable partially-executable subgoal-partially-executable; do
  # skip cache (no memory/just ask)
  create_experiment_window "ask-$teacher_type" "--teacher-type $teacher_type --memory --skip-cache --hardcoded-mode"

  # memory
  create_experiment_window "mem-$teacher_type" "--teacher-type $teacher_type --memory --hardcoded-mode"

  # memory + parse
  create_experiment_window "mem-$teacher_type-parse" "--teacher-type $teacher_type --memory --parse-teacher-answer --hardcoded-mode"

  # memory + relevance check
  create_experiment_window "mem-$teacher_type-relevance" "--teacher-type $teacher_type --memory --relevance-check --hardcoded-mode"

  # memory + parse + relevance check
  create_experiment_window "mem-$teacher_type-parse-relevance" "--teacher-type $teacher_type --memory --parse-teacher-answer --relevance-check --hardcoded-mode"
done


for teacher_type in non-executable; do
  # skip cache (no memory/just ask)
  create_experiment_window "ask-$teacher_type" "--teacher-type $teacher_type --memory  --skip-cache --clear-first"
  
  # memory
  create_experiment_window "mem-$teacher_type" "--teacher-type $teacher_type --memory --clear-first"

  # memory + parse
  create_experiment_window "mem-$teacher_type-parse" "--teacher-type $teacher_type --memory --parse-teacher-answer --clear-first"

  # memory + relevance check
  create_experiment_window "mem-$teacher_type-relevance" "--teacher-type $teacher_type --memory --relevance-check --clear-first"

  # memory + parse + relevance check
  create_experiment_window "mem-$teacher_type-parse-relevance" "--teacher-type $teacher_type --memory --parse-teacher-answer --relevance-check --clear-first"
done