#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from nano_harness.baseline import sha256_file
from nano_harness.mbpp_verified_selection import load_cases, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/campaign/mbpp_sanitized_verified_selection_dev_v1.json"
)
PUBLIC = (
    ROOT
    / "docs/experiments/"
    "mbpp_sanitized_verified_selection_dev_v1.preregister.json"
)
MARKDOWN = (
    ROOT
    / "docs/experiments/mbpp_sanitized_verified_selection_dev_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def sandbox_probe(config: dict[str, Any]) -> dict[str, bool]:
    sandbox = config["sandbox"]
    with tempfile.TemporaryDirectory() as directory:
        program = Path(directory) / "program.py"
        program.write_text(
            "from pathlib import Path\n"
            "Path('/tmp/ok').write_text('ok')\n"
            "assert Path('/tmp/ok').read_text() == 'ok'\n",
            encoding="utf-8",
        )
        sandbox_prefix = [
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
        ]
        run_program = [
            *sandbox_prefix,
            "/usr/bin/python3",
            "-I",
            "/tmp/program.py",
        ]
        writable_tmp = subprocess.run(
            run_program,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        ).returncode == 0
        network = subprocess.run(
            [
                *sandbox_prefix,
                "/usr/bin/python3",
                "-I",
                "-c",
                (
                    "import socket; "
                    "socket.create_connection(('1.1.1.1',80),1)"
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        root_write = subprocess.run(
            [
                *sandbox_prefix,
                "/usr/bin/python3",
                "-I",
                "-c",
                "open('/etc/mbpp-probe','w').write('bad')",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    return {
        "temporary_workdir_writable": writable_tmp,
        "network_blocked": network.returncode != 0,
        "root_filesystem_read_only": root_write.returncode != 0,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config(CONFIG)
    dataset = config["dataset"]
    validation_path = (ROOT / dataset["validation_path"]).resolve()
    test_path = (ROOT / dataset["test_path"]).resolve()
    if (
        sha256_file(validation_path) != dataset["validation_sha256"]
        or sha256_file(test_path) != dataset["test_sha256"]
    ):
        raise ValueError("MBPP dataset identity differs")
    cases = load_cases(config, ROOT)
    case_ids = [case.case_id for case in cases]
    probe = sandbox_probe(config)
    if not all(probe.values()):
        raise ValueError("MBPP sandbox probe failed")
    return {
        "schema_version": (
            "nano_harness_mbpp_verified_selection_preregister_v1"
        ),
        "experiment_id": config["experiment_id"],
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "validation_sha256": dataset["validation_sha256"],
            "test_sha256": dataset["test_sha256"],
            "validation_case_ids_sha256": hashlib.sha256(
                "\n".join(case_ids).encode()
            ).hexdigest(),
            "four_b_model_config_sha256": config["models"]["four_b"][
                "model_config_sha256"
            ],
            "nine_b_model_config_sha256": config["models"]["nine_b"][
                "model_config_sha256"
            ],
        },
        "surface": {
            "split": "validation",
            "cases": len(cases),
            "test_feasibility_probe_rows": 1,
            "test_content_used_for_policy_design": False,
            "test_generation_allowed": False,
        },
        "execution": config["execution"],
        "arms": {
            "four_b_direct": config["direct"],
            "nine_b_direct": config["direct"],
            "four_b_verified_selection": config["candidate"],
        },
        "sandbox": {
            **config["sandbox"],
            "probe": probe,
            "public_test_source_visible_to_model": True,
            "only_aggregate_failure_feedback_visible_to_repair": True,
        },
        "decision_rule": {
            "validation_admitted": (
                "all four_b_preservation and nine_b_directional gates pass"
            ),
            "complete_test_preregistration_allowed": "validation_admitted",
            "rerun_or_tuning_allowed": False,
        },
        "execution_boundary": config["execution_boundary"],
        "claim_boundary": (
    "This pre-registers the complete 43-case sanitized validation "
    "split. One test row was previously inspected only for local schema "
    "feasibility and did not determine this policy. It starts no model "
            "generation, and establishes no MBPP score."
        ),
    }


def render_markdown(receipt: dict[str, Any]) -> str:
    return f"""# MBPP Sanitized Verified Selection Dev v1

This pre-registers a full 43-case validation run and starts no model
generation. One test row was previously inspected only for schema feasibility;
it did not determine this policy. The 257-case sanitized test generation
remains closed.

## Candidate

- Run matched direct Qwen3.5-4B and Qwen3.5-9B once per task.
- If direct 4B passes every public test, preserve it without extra calls.
- Otherwise generate three independent 4B candidates.
- Include the public MBPP `test_list` in the model prompt, matching the
  benchmark protocol; keep reference solutions hidden.
- Execute candidates against those tests in a no-network, read-only-root
  bubblewrap sandbox.
- Select by public-test pass count, then shorter code, then replica index.
- If none passes all tests, allow one repair using only aggregate pass count
  and failure classes; reference code remains hidden.
- Override direct 4B only when the selected candidate passes more public tests.
- Run four deterministic shards assigned by sorted case index modulo four;
  merge requires the exact 43-case set.

## Identity

- config SHA: `{receipt['identity']['config_sha256']}`;
- validation data SHA: `{receipt['identity']['validation_sha256']}`;
- validation case IDs SHA:
  `{receipt['identity']['validation_case_ids_sha256']}`;
- cases: {receipt['surface']['cases']}.

## Gate

Validation must preserve 4B with a non-negative paired bootstrap lower bound
and show a positive directional delta over matched 9B with more wins than
losses. Only then may the 257-case sanitized test be separately
pre-registered. No post-observation tuning or validation rerun is allowed.
"""


def main() -> None:
    receipt = build_receipt()
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN.write_text(render_markdown(receipt), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
