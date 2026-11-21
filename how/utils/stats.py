from typing import Any

import pandas as pd
from sklearn.metrics import f1_score
import wandb

from how.datasets.plancraft import MAX_PLAN_COST

# show all columns
pd.set_option("display.max_columns", None)
pd.set_option("future.no_silent_downcasting", True)


def flatten_df(df):
    """Flatten a DataFrame with multi-level columns."""
    df.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in df.columns.values
    ]
    return df


def calculate_evaluation_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    df.sort_values(by=["order_id"], inplace=True)
    df = pd.concat([df, df.trace.apply(pd.Series)], axis=1)
    df = pd.concat([df, df.main_agent.apply(pd.Series)], axis=1)
    df = pd.concat([df, df.dialogue_stats.apply(pd.Series).fillna(0)], axis=1)
    df.drop(
        columns=[
            "trace",
            "main_agent",
            "dialogue_stats",
        ],
        inplace=True,
    )
    df["agent_total_tokens"] = df["tokens_used"].astype(int)

    # Calculate success
    success_by_complexity = (
        df.groupby("complexity").agg({"success": ["mean", "count"]}).reset_index()
    )
    wandb.log(
        {
            "success_by_complexity": wandb.Table(
                dataframe=flatten_df(success_by_complexity)
            )
        }
    )

    success_by_recipe_type = (
        df.groupby("recipe_type").agg({"success": ["mean", "count"]}).reset_index()
    )
    wandb.log(
        {
            "success_by_recipe_type": wandb.Table(
                dataframe=flatten_df(success_by_recipe_type)
            )
        }
    )

    # Calculate F1 score
    impossible_f1 = f1_score(
        df["impossible_calls"].astype(int),
        (df["complexity"] == "impossible").astype(int),
        average="macro",
    )

    if "memory_cache" in df.columns:
        df = pd.concat([df, df.memory_cache.apply(pd.Series)], axis=1)
        df = pd.concat([df, df.memory_stats.apply(pd.Series)], axis=1)
        df.drop(columns=["memory_cache", "memory_stats"], inplace=True)
        df["memory_total_tokens"] = df["total_tokens"].astype(int)
        df["miss_bucket"] = df["cache_misses"].apply(lambda x: x if x <= 2 else 3)
        df["hit_bucket"] = df["cache_hits"].apply(lambda x: x if x <= 2 else 3)
        bucket_stats = (
            df.groupby(["miss_bucket", "hit_bucket"])
            .agg({"success": ["mean", "count", "sum"]})
            .reset_index()
        )
        wandb.log({"bucket_stats": wandb.Table(dataframe=flatten_df(bucket_stats))})
    else:
        df["memory_total_tokens"] = 0

    df["total_tokens"] = df["memory_total_tokens"] + df["agent_total_tokens"]

    df["eager_crafting_error"] = df["num_actions_to_goal"].apply(
        lambda x: MAX_PLAN_COST in x
    )
    df["impossible_error"] = (df["impossible_calls"] > 0) & (~df["success"].astype(int))
    df["max_steps_error"] = df.num_steps >= 30

    overall_stats = {
        "success": df["success"].mean(),
        "avg_tokens": df["total_tokens"].mean(),
        "impossible_f1": impossible_f1,
        "total_errors": (~df["success"].astype(int)).sum(),
        "eager_crafting_error": df["eager_crafting_error"].sum(),
        "impossible_error": df["impossible_error"].sum(),
        "max_steps_error": df["max_steps_error"].sum(),
        "avg_num_steps": df["num_steps"].mean(),
    }
    wandb.log(overall_stats)
    return overall_stats
