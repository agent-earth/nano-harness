from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_harness.adapters.taubench import (
    make_tau_task,
    serialize_tau_result,
)
from nano_harness.client import OpenRouterClient
from nano_harness.config import HarnessConfig, ModelConfig
from nano_harness.harness import AgentHarness
from nano_harness.runner import append_jsonl_atomic, completed_task_ids, summarize_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["retail", "airline"], default="retail")
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategy", choices=["base", "optimized"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-split", default="test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()

    from tau_bench.envs import get_env

    probe = get_env(
        args.env,
        user_strategy="verify",
        user_model="gpt-4o",
        user_provider="openai",
        task_split=args.task_split,
    )
    end = len(probe.tasks) if args.end < 0 else min(args.end, len(probe.tasks))
    output = Path(args.output)
    completed = completed_task_ids(output)
    harness = AgentHarness(
        OpenRouterClient(ModelConfig(name=args.model)),
        args.model,
        HarnessConfig(strategy=args.strategy, max_steps=args.max_steps),
    )
    for task_index in range(args.start, end):
        if task_index % args.num_shards != args.shard_id:
            continue
        if str(task_index) in completed:
            continue
        env = get_env(
            args.env,
            user_strategy="verify",
            user_model="gpt-4o",
            user_provider="openai",
            task_split=args.task_split,
            task_index=task_index,
        )
        task, executor = make_tau_task(env, task_index, probe.wiki, probe.tools_info)
        result = harness.run(task, executor)
        append_jsonl_atomic(output, serialize_tau_result(result, executor))
    print(json.dumps(summarize_paths([output]), indent=2))


if __name__ == "__main__":
    main()
