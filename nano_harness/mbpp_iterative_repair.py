from __future__ import annotations

import ast
import hashlib
import json
import re
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from openai import OpenAI

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_verified_selection import (
    MbppCase,
    run_public_tests,
    validate_code,
)
from nano_harness.orca_self_consistency import _request


CONFIG_SHA256 = (
    "21be8ace1aa02b98f1daf7160ce5a14cc1c45138b357fd5df0ec35e51070b0df"
)
FENCED_CODE = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
BEGIN_DONE_CODE = re.compile(
    r"\[BEGIN\]\s*\n?(.*?)\n?\[DONE\]",
    re.DOTALL,
)


@dataclass(frozen=True)
class FewShot:
    task_id: int
    prompt: str
    code: str
    test_list: tuple[str, ...]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("MBPP iterative-repair config SHA differs")
    if (
        config.get("schema_version")
        != "nano_harness_mbpp_iterative_repair_v2"
        or config.get("experiment_id")
        != "mbpp-sanitized-iterative-repair-train-v2"
        or config.get("dataset", {}).get("split") != "train"
        or config.get("parser")
        != {
            "first_python_fence_anywhere": True,
            "begin_done_fallback": True,
            "plain_python_fallback": True,
            "target_blind": True,
        }
        or config.get("direct")
        != {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 768,
            "seed_base": 2026083000,
        }
        or config.get("candidate")
        != {
            "preserve_passing_direct": True,
            "replicas_after_direct_failure": 5,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 768,
            "seed_base": 2026083100,
            "selection": (
                "highest_public_test_pass_count_then_shortest_code_"
                "then_replica_index"
            ),
            "repair_rounds_after_no_full_pass": 3,
            "repair_source": "current_best_candidate",
            "repair_feedback": (
                "failed_public_test_indices_plus_failure_classes"
            ),
            "repair_temperature": 0.0,
            "repair_top_p": 1.0,
            "repair_max_tokens": 768,
            "repair_seed_base": 2026083200,
            "fallback": "frozen_four_b_direct",
        }
        or config.get("execution")
        != {
            "num_shards": 4,
            "assignment": "sorted_case_index_mod_num_shards",
            "merge_requires_exact_case_set": True,
        }
        or config.get("execution_boundary")
        != {
            "train_development_generation_started": False,
            "validation_v1_rerun": False,
            "validation_v1_result_observed": True,
            "validation_rows_loaded_by_v2": False,
            "test_feasibility_probe_rows": 1,
            "test_content_used_for_policy_design": False,
            "test_generation_started": False,
            "this_commit_only_preregisters": True,
            "training_started": False,
            "rl_or_opd_started": False,
        }
        or config.get("policy")
        != {
            "reference_solution_used": False,
            "public_test_source_visible_to_model": True,
            "test_outcome_used_by_verifier": True,
            "train_rows_training_eligible": False,
            "validation_rows_training_eligible": False,
            "test_rows_training_eligible": False,
            "outputs_may_enter_training_reward_or_verifier": False,
            "raw_outputs_committed": False,
            "post_observation_tuning": False,
        }
    ):
        raise ValueError("MBPP iterative-repair contract differs")
    return config


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in Dataset.from_parquet(str(path))]


def load_train_cases(
    config: dict[str, Any],
    root: Path,
) -> list[MbppCase]:
    dataset = config["dataset"]
    path = (root / dataset["train_path"]).resolve()
    if sha256_file(path) != dataset["train_sha256"]:
        raise ValueError("MBPP train dataset differs")
    rows = _load_dataset(path)
    cases = [
        MbppCase(
            case_id=f"mbpp-sanitized-train-{int(row['task_id'])}",
            task_id=int(row["task_id"]),
            prompt=str(row["prompt"]),
            test_imports=tuple(str(value) for value in row["test_imports"]),
            test_list=tuple(str(value) for value in row["test_list"]),
        )
        for row in rows
    ]
    if len(cases) != dataset["train_rows"]:
        raise ValueError("MBPP train row count differs")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("MBPP train case IDs are duplicated")
    return sorted(cases, key=lambda case: case.case_id)


def load_few_shots(
    config: dict[str, Any],
    root: Path,
) -> list[FewShot]:
    dataset = config["dataset"]
    path = (root / dataset["prompt_path"]).resolve()
    if sha256_file(path) != dataset["prompt_sha256"]:
        raise ValueError("MBPP prompt dataset differs")
    wanted = dataset["few_shot_task_ids"]
    by_id = {int(row["task_id"]): row for row in _load_dataset(path)}
    if any(task_id not in by_id for task_id in wanted):
        raise ValueError("MBPP few-shot identity differs")
    return [
        FewShot(
            task_id=task_id,
            prompt=str(by_id[task_id]["prompt"]),
            code=str(by_id[task_id]["code"]).strip(),
            test_list=tuple(
                str(value) for value in by_id[task_id]["test_list"]
            ),
        )
        for task_id in wanted
    ]


def extract_code(output: str) -> str | None:
    fenced = FENCED_CODE.search(output)
    if fenced is not None:
        candidate = fenced.group(1).strip()
        if candidate:
            return candidate
    begin_done = BEGIN_DONE_CODE.search(output)
    if begin_done is not None:
        candidate = begin_done.group(1).strip()
        if candidate:
            return candidate
    candidate = output.strip()
    if not candidate:
        return None
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return None
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in tree.body
    ):
        return None
    return candidate


def task_messages(
    config: dict[str, Any],
    case: MbppCase,
    few_shots: list[FewShot],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": config["prompt"]["system"]}
    ]
    for example in few_shots:
        tests = "\n".join(example.test_list)
        messages.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f"Task: {example.prompt}\n\n"
                        f"Tests:\n{tests}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"```python\n{example.code}\n```",
                },
            ]
        )
    tests = "\n".join(case.test_list)
    messages.append(
        {
            "role": "user",
            "content": f"Task: {case.prompt}\n\nTests:\n{tests}",
        }
    )
    return messages


def request_code(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    reply = _request(
        client,
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        parser=extract_code,
    )
    reply["code"] = reply.pop("prediction")
    return reply


def select_best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("MBPP iterative candidate list is empty")
    return min(
        candidates,
        key=lambda row: (
            -row["test_result"]["passed"],
            len(row["code"] or ""),
            row["candidate_index"],
        ),
    )


def repair_messages(
    config: dict[str, Any],
    case: MbppCase,
    few_shots: list[FewShot],
    best: dict[str, Any],
) -> list[dict[str, str]]:
    messages = task_messages(config, case, few_shots)
    feedback = {
        "passed": best["test_result"]["passed"],
        "total": best["test_result"]["total"],
        "failed_public_test_indices": best["test_result"].get(
            "failed_test_indices", []
        ),
        "failure_classes": best["test_result"]["failure_classes"],
    }
    messages[-1] = {
        "role": "user",
        "content": (
            f"{messages[-1]['content']}\n\n"
            "Current candidate:\n"
            f"```python\n{best['code'] or ''}\n```\n\n"
            f"Public-test feedback: {json.dumps(feedback, sort_keys=True)}\n"
            "Repair the code and return one Python code block."
        ),
    }
    return messages


def generate_case(
    config: dict[str, Any],
    case: MbppCase,
    few_shots: list[FewShot],
    *,
    four_client: Any,
    nine_client: Any,
    case_index: int,
) -> dict[str, Any]:
    direct_config = config["direct"]
    candidate_config = config["candidate"]
    messages = task_messages(config, case, few_shots)
    direct = {}
    for name, client in (("four_b", four_client), ("nine_b", nine_client)):
        model = config["models"][name]["model"]
        offset = 0 if name == "four_b" else 9
        reply = request_code(
            client,
            model=model,
            messages=messages,
            seed=direct_config["seed_base"] + case_index * 10 + offset,
            temperature=direct_config["temperature"],
            top_p=direct_config["top_p"],
            max_tokens=direct_config["max_tokens"],
        )
        direct[name] = {
            **reply,
            "test_result": run_public_tests(
                reply["code"], case, config["sandbox"]
            ),
        }

    candidate = direct["four_b"]
    generated: list[dict[str, Any]] = []
    receipt: dict[str, Any] = {
        "direct_full_pass": direct["four_b"]["test_result"]["full_pass"],
        "replicas_generated": 0,
        "repair_rounds_generated": 0,
        "selected_source": "four_b_direct",
        "override": False,
    }
    if not direct["four_b"]["test_result"]["full_pass"]:
        for replica_index in range(
            candidate_config["replicas_after_direct_failure"]
        ):
            reply = request_code(
                four_client,
                model=config["models"]["four_b"]["model"],
                messages=messages,
                seed=(
                    candidate_config["seed_base"]
                    + case_index * 10
                    + replica_index
                ),
                temperature=candidate_config["temperature"],
                top_p=candidate_config["top_p"],
                max_tokens=candidate_config["max_tokens"],
            )
            generated.append(
                {
                    **reply,
                    "candidate_index": replica_index,
                    "source": f"replica_{replica_index}",
                    "test_result": run_public_tests(
                        reply["code"], case, config["sandbox"]
                    ),
                }
            )
        receipt["replicas_generated"] = len(generated)
        best = select_best(
            [
                {
                    **direct["four_b"],
                    "candidate_index": -1,
                    "source": "four_b_direct",
                },
                *generated,
            ]
        )
        for repair_round in range(
            candidate_config["repair_rounds_after_no_full_pass"]
        ):
            if best["test_result"]["full_pass"]:
                break
            reply = request_code(
                four_client,
                model=config["models"]["four_b"]["model"],
                messages=repair_messages(config, case, few_shots, best),
                seed=(
                    candidate_config["repair_seed_base"]
                    + case_index * 10
                    + repair_round
                ),
                temperature=candidate_config["repair_temperature"],
                top_p=candidate_config["repair_top_p"],
                max_tokens=candidate_config["repair_max_tokens"],
            )
            repair = {
                **reply,
                "candidate_index": len(generated),
                "source": f"repair_{repair_round}",
                "test_result": run_public_tests(
                    reply["code"], case, config["sandbox"]
                ),
            }
            generated.append(repair)
            best = select_best([best, repair])
            receipt["repair_rounds_generated"] += 1
        if (
            best["test_result"]["passed"]
            > direct["four_b"]["test_result"]["passed"]
        ):
            candidate = best
            receipt["selected_source"] = best["source"]
            receipt["override"] = True
        receipt["best_generated_test_result"] = best["test_result"]
    return {
        "schema_version": "nano_harness_mbpp_iterative_repair_case_v2",
        "case_id": case.case_id,
        "task_id": case.task_id,
        "four_b_direct": direct["four_b"],
        "nine_b_direct": direct["nine_b"],
        "candidate": {
            "output": candidate["output"],
            "code": candidate["code"],
            "test_result": candidate["test_result"],
        },
        "receipt": receipt,
    }


def select_shard(
    cases: list[MbppCase],
    *,
    num_shards: int,
    shard_id: int,
) -> list[tuple[int, MbppCase]]:
    if num_shards <= 0 or shard_id not in range(num_shards):
        raise ValueError("MBPP iterative-repair shard differs")
    return [
        (index, case)
        for index, case in enumerate(cases)
        if index % num_shards == shard_id
    ]


def verify_services(config: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, model in config["models"].items():
        with urllib.request.urlopen(
            model["base_url"] + "/models",
            timeout=30,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("data", [])
        if (
            len(rows) != 1
            or rows[0].get("id") != model["model"]
            or rows[0].get("max_model_len") != model["max_model_len"]
            or rows[0].get("owned_by") != "vllm"
        ):
            raise ValueError(f"MBPP v2 {name} service identity differs")
        result[name] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(
    config_path: str | Path,
    *,
    num_shards: int,
    shard_id: int,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = load_config(config_path)
    predecessor = root / config["predecessor"]["v1_report_path"]
    if (
        sha256_file(predecessor)
        != config["predecessor"]["v1_report_sha256"]
        or json.loads(predecessor.read_text(encoding="utf-8"))
        .get("decision", {})
        .get("validation_admitted")
        is not config["predecessor"]["v1_admitted"]
    ):
        raise ValueError("MBPP v2 predecessor differs")
    for key in ("validation_path", "test_path"):
        path = (root / config["dataset"][key]).resolve()
        digest = config["dataset"][key.removesuffix("_path") + "_sha256"]
        if sha256_file(path) != digest:
            raise ValueError(f"MBPP v2 {key} identity differs")
    cases = load_train_cases(config, root)
    few_shots = load_few_shots(config, root)
    if (
        num_shards != config["execution"]["num_shards"]
        or config["execution"]["assignment"]
        != "sorted_case_index_mod_num_shards"
        or config["execution"]["merge_requires_exact_case_set"] is not True
    ):
        raise ValueError("MBPP v2 execution contract differs")
    selected = select_shard(
        cases,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    service_sha = verify_services(config)
    output_root = root / config["output_dir"]
    output_path = output_root / f"shard-{shard_id}.jsonl"
    completed = {row["case_id"] for row in read_jsonl(output_path)}
    expected_ids = {case.case_id for _, case in selected}
    if not completed.issubset(expected_ids):
        raise ValueError("MBPP v2 output IDs differ")
    output_root.mkdir(parents=True, exist_ok=True)
    clients = {
        name: OpenAI(
            api_key="local-vllm",
            base_url=model["base_url"],
            timeout=240,
            max_retries=0,
        )
        for name, model in config["models"].items()
    }
    started = time.time()
    for case_index, case in selected:
        if case.case_id in completed:
            continue
        row = generate_case(
            config,
            case,
            few_shots,
            four_client=clients["four_b"],
            nine_client=clients["nine_b"],
            case_index=case_index,
        )
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    rows = read_jsonl(output_path)
    if (
        len(rows) != len(selected)
        or {row["case_id"] for row in rows} != expected_ids
    ):
        raise ValueError("MBPP v2 generation incomplete")
    result = {
        "schema_version": "nano_harness_mbpp_iterative_repair_raw_v2",
        "experiment_id": config["experiment_id"],
        "identity": {
            "config_sha256": sha256_file(config_path),
            "raw_sha256": sha256_file(output_path),
            "service_sha256": service_sha,
        },
        "surface": {
            "split": config["dataset"]["split"],
            "cases": len(rows),
            "num_shards": num_shards,
            "shard_id": shard_id,
            "validation_v1_rerun": False,
            "validation_rows_loaded_by_v2": False,
            "test_generation_started": False,
        },
        "wall_seconds": time.time() - started,
    }
    (output_root / f"shard-{shard_id}.result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
