#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from nano_harness.baseline import sha256_file
from nano_harness.semantic_skill_execution import (
    FAMILIES,
    SKILL_PROMPTS,
    build_cases as build_parent_cases,
    load_config as load_parent_config,
    route_prompt,
)
from nano_harness.semantic_skill_replication import (
    build_cases,
    load_config,
    parent_config,
)
from nano_harness.verified_tool_execution import (
    contamination_audit,
    public_case_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/harness/qwen35_semantic_skill_replication_v1.json"
)
JSON_OUTPUT = (
    ROOT
    / "docs/experiments/qwen35_semantic_skill_replication_v1.preregister.json"
)
MARKDOWN_OUTPUT = (
    ROOT / "docs/experiments/qwen35_semantic_skill_replication_v1.md"
)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _normalized_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def build_receipt() -> dict:
    config = load_config(CONFIG)
    mechanism = load_parent_config(config.parent_config_path)
    parent = parent_config(config)
    cases = build_cases(config)
    parent_cases = build_parent_cases(mechanism)
    contract = public_case_contract(cases)
    audit = contamination_audit(parent, cases)
    if not audit["passed"]:
        raise ValueError("semantic replication contamination audit failed")

    case_ids = {row["case_id"] for row in cases}
    parent_case_ids = {row["case_id"] for row in parent_cases}
    prompt_hashes = {_normalized_hash(row["prompt"]) for row in cases}
    parent_prompt_hashes = {
        _normalized_hash(row["prompt"]) for row in parent_cases
    }
    source_hashes = {
        hashlib.sha256(
            json.dumps(
                row["source_facts"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for row in cases
    }
    parent_source_hashes = {
        hashlib.sha256(
            json.dumps(
                row["source_facts"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for row in parent_cases
    }
    routes = [route_prompt(case["prompt"]) for case in cases]
    if (
        len(cases) != 256
        or case_ids & parent_case_ids
        or prompt_hashes & parent_prompt_hashes
        or source_hashes & parent_source_hashes
        or any(not route["routed"] for route in routes)
        or any(route["router_uses_case_metadata"] for route in routes)
    ):
        raise ValueError("semantic replication freshness or routing differs")

    tokenizer = AutoTokenizer.from_pretrained(
        parent.four_b_model_path,
        local_files_only=True,
    )
    plan_lengths = {}
    for family in FAMILIES:
        selected = [case for case in cases if case["family"] == family]
        plan_lengths[family] = max(
            len(
                tokenizer.encode(
                    tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": SKILL_PROMPTS[family],
                            },
                            {"role": "user", "content": case["prompt"]},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        **parent.chat_template_kwargs,
                    )
                )
            )
            for case in selected
        )
    if max(plan_lengths.values()) + config.plan_max_tokens > parent.max_model_len:
        raise ValueError("semantic replication context budget differs")

    parent_report = json.loads(
        Path(config.parent_report_path).read_text(encoding="utf-8")
    )
    return {
        "schema_version": (
            "nano_harness_semantic_skill_replication_preregister_v1"
        ),
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "parent_config_sha256": config.parent_config_sha256,
            "parent_preregister_sha256": config.parent_preregister_sha256,
            "parent_report_sha256": config.parent_report_sha256,
            "case_contract_sha256": contract["case_contract_sha256"],
        },
        "case_contract": contract,
        "freshness": {
            "parent_case_id_overlap": len(case_ids & parent_case_ids),
            "parent_prompt_overlap": len(prompt_hashes & parent_prompt_hashes),
            "parent_source_fact_overlap": len(
                source_hashes & parent_source_hashes
            ),
            "prior_surface_prompt_overlap": audit[
                "prior_surface_prompt_overlap"
            ],
            "benchmark_prompt_overlap": audit["benchmark_prompt_overlap"],
            "benchmark_rows_hashed": audit["benchmark_rows_hashed"],
            "benchmark_outputs_loaded": False,
            "canary_rows_or_outputs_loaded": False,
            "holdout_rows_loaded": False,
        },
        "mechanism_invariance": {
            "skill_router": mechanism.skill_router,
            "route_markers": mechanism.route_markers,
            "plan_structured_output_regex_by_family": (
                mechanism.plan_structured_output_regex_by_family
            ),
            "direct_max_tokens": config.direct_max_tokens,
            "plan_max_tokens": config.plan_max_tokens,
            "final_max_tokens": config.final_max_tokens,
            "plan_retry_limit": config.plan_retry_limit,
            "temperature": parent.temperature,
            "chat_template_kwargs": parent.chat_template_kwargs,
            "bootstrap_samples": config.bootstrap_samples,
            "significance_alpha": config.significance_alpha,
            "minimum_harness_wins": config.minimum_harness_wins,
            "maximum_harness_losses": config.maximum_harness_losses,
        },
        "replication_delta": {
            "retained": [
                "prompt-marker router and exact route markers",
                "single applicable skill exposure",
                "typed tool schemas and semantic executor",
                "source-fact validation",
                "one retry and direct fallback",
                "verified-result feedback and equality check",
                "models, services, temperature, budgets, and gates",
            ],
            "changed": [
                "unseen compact-display and kiosk contexts",
                "small integer numerical regime",
                "fresh case seed and case identities",
            ],
            "parent_result": {
                "harness_correct": parent_report["arms"][
                    "four_b_semantic_skills"
                ]["correct"],
                "four_b_direct_correct": parent_report["arms"][
                    "four_b_direct"
                ]["correct"],
                "nine_b_direct_correct": parent_report["arms"][
                    "nine_b_direct"
                ]["correct"],
                "local_semantic_skill_admitted": parent_report["decision"][
                    "local_semantic_skill_admitted"
                ],
            },
        },
        "token_budget": {
            "maximum_plan_input_tokens_by_family": plan_lengths,
            "max_model_len": parent.max_model_len,
            "plan_max_tokens": config.plan_max_tokens,
        },
        "acceptance": {
            "cases": 256,
            "cases_per_family": 128,
            "all_rows_complete_and_parseable": True,
            "prompt_routes_256": True,
            "single_tool_exposures_256": True,
            "verified_executions_256": True,
            "feedback_result_matches_256": True,
            "executor_contract_failures_zero": True,
            "harness_vs_four_b_significant": True,
            "harness_vs_four_b_minimum_wins": config.minimum_harness_wins,
            "harness_vs_four_b_maximum_losses": config.maximum_harness_losses,
            "harness_vs_nine_b_significant": True,
            "harness_vs_nine_b_minimum_wins": config.minimum_harness_wins,
            "harness_vs_nine_b_maximum_losses": config.maximum_harness_losses,
            "every_family_non_regression_vs_four_b_and_nine_b": True,
            "real_task_preregistration_allowed_after_pass": True,
            "canary_rerun_allowed_after_pass": False,
            "benchmark_generation_allowed_after_pass": False,
            "holdout_allowed_after_pass": False,
            "training_allowed_after_pass": False,
        },
        "decision_policy": {
            "forbidden_after_observation": [
                "case_change",
                "prompt_template_change",
                "numerical_regime_change",
                "route_marker_change",
                "skill_schema_change",
                "semantic_executor_change",
                "prompt_change",
                "regex_change",
                "retry_change",
                "budget_change",
                "temperature_change",
                "model_or_service_change",
                "fallback_change",
                "gate_change",
                "arm_rerun",
                "canary_access",
                "benchmark_access",
                "holdout_access",
            ],
            "passed": (
                "Publish replication evidence and separately pre-register one "
                "real-task transfer without reopening the observed canary. "
                "Generation remains closed until that preregistration commits."
            ),
            "failed": (
                "Publish negative evidence and stop this mechanism. Do not tune "
                "or rerun on the observed replication surface."
            ),
        },
        "execution_boundary": config.execution_boundary,
        "claim_boundary": (
            "This pre-registers a fresh local replication only. It generates "
            "no model output and does not access benchmark, canary, holdout, "
            "or training data."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Typed Semantic Skill Replication v1

## 唯一变化

这次保持 parent semantic-skill mechanism 完全不变，只换证据分布：

- 新场景：compact display / kiosk forecast；
- 新数值范围：中小整数，不再使用 parent 的超大 offset；
- 新 case seed 和 case IDs。

Router markers、skill prompts、tool schemas、semantic executor、source validator、
one retry、direct fallback、result feedback、feedback equality、models、services、
temperature、budgets 和统计 gate 都不变。

## Freshness

- 256 cases，2 families × 128；
- case contract SHA：
  `{receipt['identity']['case_contract_sha256']}`；
- parent case ID overlap：0；
- parent prompt overlap：0；
- parent source-fact overlap：0；
- prior choice/tool prompt overlap：0；
- complete GSM8K/MMLU/GPQA prompt overlap：0；
- benchmark/canary/holdout outputs 或 rows：0。

## Gate

- 256/256 complete and parseable；
- 256 prompt routes、single-tool exposures、verified executions、feedback matches；
- 0 retry/fallback/contract failure；
- harness vs 4B 和 vs 9B 均 CI lower > 0、McNemar p < 0.05、
  至少 12 wins、0 losses；
- every-family non-regression。

通过也只允许另行预注册 real-task transfer；不允许重跑已观察 canary，不允许
直接生成 complete benchmark，不开放 independent holdout 或 training。

## Boundaries

- config SHA：`{receipt['identity']['config_sha256']}`；
- parent report SHA：`{receipt['identity']['parent_report_sha256']}`；
- model generation：false；
- evaluation started：false；
- canary / benchmark / holdout accessed：false；
- training started：false。

观察结果后禁止修改模板、数值范围、markers、schema、executor、prompt、
regex、retry、budget、fallback、gate 或重跑。
"""


def main() -> None:
    receipt = build_receipt()
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT.write_text(
        render_markdown(receipt),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiment_id": receipt["experiment_id"],
                "identity": receipt["identity"],
                "freshness": receipt["freshness"],
                "mechanism_invariance": receipt["mechanism_invariance"],
                "acceptance": receipt["acceptance"],
                "execution_boundary": receipt["execution_boundary"],
                "json_output": str(JSON_OUTPUT),
                "markdown_output": str(MARKDOWN_OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
