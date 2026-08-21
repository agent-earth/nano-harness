from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from openai import OpenAI

from nano_harness.baseline import sha256_file
from nano_harness.orca_self_consistency import _request


CONFIG_SHA256 = (
    "8e02b0adc5ca78a4a197ec6622bcc65a0b30de3b6f9cdb64aa6446ce453775c2"
)
CODE_BLOCK = re.compile(
    r"\A\s*```(?:python)?\s*\n(.*?)\n```\s*\Z",
    re.DOTALL | re.IGNORECASE,
)
DANGEROUS_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
DANGEROUS_ATTRIBUTES = {
    "argv",
    "executable",
    "modules",
    "path",
    "setprofile",
    "settrace",
    "stderr",
    "stdin",
    "stdout",
}


@dataclass(frozen=True)
class MbppCase:
    case_id: str
    task_id: int
    prompt: str
    test_imports: tuple[str, ...]
    test_list: tuple[str, ...]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("MBPP verified-selection config SHA differs")
    if (
        config.get("schema_version")
        != "nano_harness_mbpp_verified_selection_v1"
        or config.get("experiment_id")
        != "mbpp-sanitized-verified-selection-dev-v1"
        or config.get("dataset", {}).get("split") != "validation"
        or config.get("candidate")
        != {
            "preserve_passing_direct": True,
            "replicas_after_direct_failure": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 768,
            "seed_base": 2026082800,
            "selection": (
                "highest_public_test_pass_count_then_shortest_code_"
                "then_replica_index"
            ),
            "repair_after_no_full_pass": True,
            "repair_source": "best_candidate",
            "repair_feedback": (
                "aggregate_pass_count_and_failure_classes_only"
            ),
            "repair_temperature": 0.0,
            "repair_top_p": 1.0,
            "repair_max_tokens": 768,
            "repair_seed_base": 2026082900,
            "fallback": "frozen_four_b_direct",
        }
        or config.get("direct")
        != {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 768,
            "seed_base": 2026082700,
        }
        or config.get("sandbox")
        != {
            "backend": "bubblewrap",
            "network": False,
            "root_filesystem": "read_only",
            "temporary_home": True,
            "temporary_workdir": True,
            "python_isolated_mode": True,
            "timeout_seconds_per_test": 5,
            "cpu_seconds_per_test": 3,
            "address_space_bytes": 1_073_741_824,
            "file_size_bytes": 1_048_576,
            "open_files": 64,
            "allowed_imports": [
                "array",
                "bisect",
                "cmath",
                "collections",
                "copy",
                "decimal",
                "fractions",
                "functools",
                "heapq",
                "itertools",
                "math",
                "operator",
                "re",
                "statistics",
                "string",
                "sys",
                "typing",
            ],
        }
        or config.get("execution")
        != {
            "num_shards": 4,
            "assignment": "sorted_case_index_mod_num_shards",
            "merge_requires_exact_case_set": True,
        }
        or config.get("policy")
        != {
            "reference_solution_used": False,
            "public_test_source_visible_to_model": True,
            "test_outcome_used_by_verifier": True,
            "validation_rows_training_eligible": False,
            "test_rows_training_eligible": False,
            "outputs_may_enter_training_reward_or_verifier": False,
            "raw_outputs_committed": False,
            "post_observation_tuning": False,
        }
        or config.get("execution_boundary")
        != {
            "validation_generation_started": False,
            "test_feasibility_probe_rows": 1,
            "test_content_used_for_policy_design": False,
            "test_generation_started": False,
            "this_commit_only_preregisters": True,
            "training_started": False,
            "rl_or_opd_started": False,
        }
    ):
        raise ValueError("MBPP verified-selection contract differs")
    return config


def load_cases(
    config: dict[str, Any],
    root: Path,
) -> list[MbppCase]:
    dataset = config["dataset"]
    path = (root / dataset["validation_path"]).resolve()
    if sha256_file(path) != dataset["validation_sha256"]:
        raise ValueError("MBPP validation dataset differs")
    rows = Dataset.from_parquet(str(path))
    cases = [
        MbppCase(
            case_id=f"mbpp-sanitized-validation-{int(row['task_id'])}",
            task_id=int(row["task_id"]),
            prompt=str(row["prompt"]),
            test_imports=tuple(str(value) for value in row["test_imports"]),
            test_list=tuple(str(value) for value in row["test_list"]),
        )
        for row in rows
    ]
    if len(cases) != dataset["validation_rows"]:
        raise ValueError("MBPP validation row count differs")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("MBPP validation case IDs are duplicated")
    return sorted(cases, key=lambda case: case.case_id)


def extract_code(output: str) -> str | None:
    match = CODE_BLOCK.fullmatch(output)
    if match is None:
        return None
    code = match.group(1).strip()
    return code or None


def validate_code(code: str, allowed_imports: set[str]) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "syntax_error"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".", 1)[0] for alias in node.names]
            if any(module not in allowed_imports for module in modules):
                return "forbidden_import"
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if node.level or module not in allowed_imports:
                return "forbidden_import"
        elif isinstance(node, ast.Name) and node.id in DANGEROUS_NAMES:
            return "forbidden_builtin"
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr in DANGEROUS_ATTRIBUTES
        ):
            return "forbidden_attribute"
    return None


def run_public_tests(
    code: str | None,
    case: MbppCase,
    sandbox: dict[str, Any],
) -> dict[str, Any]:
    if code is None:
        return {
            "passed": 0,
            "total": len(case.test_list),
            "failure_classes": {"format_error": len(case.test_list)},
            "failed_test_indices": list(range(len(case.test_list))),
            "full_pass": False,
        }
    static_failure = validate_code(code, set(sandbox["allowed_imports"]))
    if static_failure is not None:
        return {
            "passed": 0,
            "total": len(case.test_list),
            "failure_classes": {static_failure: len(case.test_list)},
            "failed_test_indices": list(range(len(case.test_list))),
            "full_pass": False,
        }
    passed = 0
    failures: Counter[str] = Counter()
    failed_test_indices = []
    for test_index, test in enumerate(case.test_list):
        source = "\n".join((*case.test_imports, code, test)) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            program = Path(directory) / "program.py"
            program.write_text(source, encoding="utf-8")
            command = [
                "prlimit",
                f"--cpu={sandbox['cpu_seconds_per_test']}",
                f"--as={sandbox['address_space_bytes']}",
                f"--fsize={sandbox['file_size_bytes']}",
                f"--nofile={sandbox['open_files']}",
                "--",
                "bwrap",
                "--unshare-net",
                "--unshare-pid",
                "--die-with-parent",
                "--new-session",
                "--ro-bind",
                "/usr/bin/python3",
                "/usr/bin/python3",
                "--ro-bind",
                "/usr/lib",
                "/usr/lib",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind",
                "/lib64",
                "/lib64",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--tmpfs",
                "/home",
                "--ro-bind",
                str(program),
                "/tmp/program.py",
                "--chdir",
                "/tmp",
                "/usr/bin/python3",
                "-I",
                "/tmp/program.py",
            ]
            try:
                process = subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=sandbox["timeout_seconds_per_test"],
                    check=False,
                )
            except subprocess.TimeoutExpired:
                failures["timeout"] += 1
                failed_test_indices.append(test_index)
                continue
        if process.returncode == 0:
            passed += 1
        elif "AssertionError" in process.stderr:
            failures["assertion_failure"] += 1
            failed_test_indices.append(test_index)
        elif "SyntaxError" in process.stderr:
            failures["syntax_error"] += 1
            failed_test_indices.append(test_index)
        elif "MemoryError" in process.stderr:
            failures["memory_limit"] += 1
            failed_test_indices.append(test_index)
        else:
            failures["runtime_error"] += 1
            failed_test_indices.append(test_index)
    return {
        "passed": passed,
        "total": len(case.test_list),
        "failure_classes": dict(sorted(failures.items())),
        "failed_test_indices": failed_test_indices,
        "full_pass": passed == len(case.test_list),
    }


def task_messages(config: dict[str, Any], case: MbppCase) -> list[dict[str, str]]:
    tests = "\n".join(case.test_list)
    return [
        {"role": "system", "content": config["prompt"]["system"]},
        {
            "role": "user",
            "content": (
                f"Task: {case.prompt}\n\n"
                f"Tests:\n{tests}"
            ),
        },
    ]


def repair_messages(
    config: dict[str, Any],
    case: MbppCase,
    code: str,
    result: dict[str, Any],
) -> list[dict[str, str]]:
    tests = "\n".join(case.test_list)
    feedback = json.dumps(
        {
            "passed": result["passed"],
            "total": result["total"],
            "failure_classes": result["failure_classes"],
        },
        sort_keys=True,
    )
    return [
        {"role": "system", "content": config["prompt"]["system"]},
        {
            "role": "user",
            "content": (
                f"Task: {case.prompt}\n\n"
                f"Tests:\n{tests}\n\n"
                "Your previous candidate was:\n"
                f"```python\n{code}\n```\n\n"
                f"Hidden-test aggregate: {feedback}\n"
                "Repair the implementation using the provided tests."
            ),
        },
    ]


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
        raise ValueError("MBPP candidate list is empty")
    return min(
        candidates,
        key=lambda row: (
            -row["test_result"]["passed"],
            len(row["code"] or ""),
            row["replica_index"],
        ),
    )


def generate_case(
    config: dict[str, Any],
    case: MbppCase,
    *,
    four_client: Any,
    nine_client: Any,
    case_index: int,
) -> dict[str, Any]:
    direct = config["direct"]
    candidate_config = config["candidate"]
    common_messages = task_messages(config, case)
    four = request_code(
        four_client,
        model=config["models"]["four_b"]["model"],
        messages=common_messages,
        seed=direct["seed_base"] + case_index * 10,
        temperature=direct["temperature"],
        top_p=direct["top_p"],
        max_tokens=direct["max_tokens"],
    )
    nine = request_code(
        nine_client,
        model=config["models"]["nine_b"]["model"],
        messages=common_messages,
        seed=direct["seed_base"] + case_index * 10 + 9,
        temperature=direct["temperature"],
        top_p=direct["top_p"],
        max_tokens=direct["max_tokens"],
    )
    four_test = run_public_tests(four["code"], case, config["sandbox"])
    nine_test = run_public_tests(nine["code"], case, config["sandbox"])
    candidate = four
    candidate_test = four_test
    receipt: dict[str, Any] = {
        "direct_full_pass": four_test["full_pass"],
        "replicas_generated": 0,
        "repair_generated": False,
        "selected_source": "four_b_direct",
        "override": False,
    }
    if not four_test["full_pass"]:
        replicas = []
        for replica_index in range(
            candidate_config["replicas_after_direct_failure"]
        ):
            reply = request_code(
                four_client,
                model=config["models"]["four_b"]["model"],
                messages=common_messages,
                seed=(
                    candidate_config["seed_base"]
                    + case_index * 10
                    + replica_index
                ),
                temperature=candidate_config["temperature"],
                top_p=candidate_config["top_p"],
                max_tokens=candidate_config["max_tokens"],
            )
            replicas.append(
                {
                    **reply,
                    "replica_index": replica_index,
                    "test_result": run_public_tests(
                        reply["code"],
                        case,
                        config["sandbox"],
                    ),
                }
            )
        best = select_best(replicas)
        receipt["replicas_generated"] = len(replicas)
        if not best["test_result"]["full_pass"]:
            repair = request_code(
                four_client,
                model=config["models"]["four_b"]["model"],
                messages=repair_messages(
                    config,
                    case,
                    best["code"] or "",
                    best["test_result"],
                ),
                seed=candidate_config["repair_seed_base"] + case_index,
                temperature=candidate_config["repair_temperature"],
                top_p=candidate_config["repair_top_p"],
                max_tokens=candidate_config["repair_max_tokens"],
            )
            repair_candidate = {
                **repair,
                "replica_index": len(replicas),
                "test_result": run_public_tests(
                    repair["code"],
                    case,
                    config["sandbox"],
                ),
            }
            receipt["repair_generated"] = True
            best = select_best([best, repair_candidate])
        if best["test_result"]["passed"] > four_test["passed"]:
            candidate = best
            candidate_test = best["test_result"]
            receipt["selected_source"] = (
                "repair"
                if best["replica_index"] == len(replicas)
                else f"replica_{best['replica_index']}"
            )
            receipt["override"] = True
        receipt["best_generated_test_result"] = best["test_result"]
    return {
        "schema_version": "nano_harness_mbpp_verified_selection_case_v1",
        "case_id": case.case_id,
        "task_id": case.task_id,
        "four_b_direct": {
            **four,
            "test_result": four_test,
        },
        "nine_b_direct": {
            **nine,
            "test_result": nine_test,
        },
        "candidate": {
            "output": candidate["output"],
            "code": candidate["code"],
            "test_result": candidate_test,
        },
        "receipt": receipt,
    }


def verify_services(config: dict[str, Any]) -> dict[str, Any]:
    results = {}
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
            raise ValueError(f"MBPP {name} service identity differs")
        results[name] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return results


def select_shard(
    cases: list[MbppCase],
    *,
    num_shards: int,
    shard_id: int,
) -> list[tuple[int, MbppCase]]:
    if num_shards <= 0 or shard_id not in range(num_shards):
        raise ValueError("MBPP shard identity differs")
    return [
        (index, case)
        for index, case in enumerate(cases)
        if index % num_shards == shard_id
    ]


def run(
    config_path: str | Path,
    *,
    num_shards: int,
    shard_id: int,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = load_config(config_path)
    service_sha = verify_services(config)
    cases = load_cases(config, root)
    if (
        num_shards != config["execution"]["num_shards"]
        or config["execution"]["assignment"]
        != "sorted_case_index_mod_num_shards"
        or config["execution"]["merge_requires_exact_case_set"] is not True
    ):
        raise ValueError("MBPP execution contract differs")
    selected = select_shard(
        cases,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    output_root = root / config["output_dir"]
    output_path = output_root / f"shard-{shard_id}.jsonl"
    completed = set()
    if output_path.exists():
        for row in jsonl_rows(output_path):
            completed.add(str(row["case_id"]))
    expected_ids = {case.case_id for _, case in selected}
    if not completed.issubset(expected_ids):
        raise ValueError("MBPP output case IDs differ")
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
            four_client=clients["four_b"],
            nine_client=clients["nine_b"],
            case_index=case_index,
        )
        row["service_sha256"] = service_sha
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    rows = jsonl_rows(output_path)
    if (
        len(rows) != len(selected)
        or {row["case_id"] for row in rows} != expected_ids
    ):
        raise ValueError("MBPP validation generation incomplete")
    result = {
        "schema_version": "nano_harness_mbpp_verified_selection_raw_v1",
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
            "test_feasibility_probe_rows": 1,
            "test_content_used_for_policy_design": False,
        },
        "wall_seconds": time.time() - started,
        "evaluation_boundary": {
            "public_tests_visible_to_model": True,
            "reference_solution_used": False,
            "test_outcome_used_by_verifier": True,
            "benchmark_rows_training_eligible": False,
            "raw_outputs_committed": False,
        },
    }
    (output_root / f"shard-{shard_id}.result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
