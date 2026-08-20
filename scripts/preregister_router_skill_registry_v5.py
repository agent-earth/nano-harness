#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from nano_harness.baseline import sha256_file
from nano_harness.router_adapter_integration import ROUTER_SYSTEM
from nano_harness.router_adapter_integration import build_cases as build_v1
from nano_harness.router_adapter_integration import load_config as load_v1
from nano_harness.router_adapter_integration_v2 import build_cases as build_v2
from nano_harness.router_adapter_integration_v2 import load_config as load_v2
from nano_harness.router_adapter_integration_v3 import build_cases as build_v3
from nano_harness.router_adapter_integration_v3 import load_config as load_v3
from nano_harness.router_skill_fallback_v4 import build_cases as build_v4
from nano_harness.router_skill_fallback_v4 import load_config as load_v4
from nano_harness.router_skill_registry_v5 import (
    C_FAMILIES,
    POSITIVE_FAMILIES,
    SKILL_PROMPTS,
    SKILL_REGEX,
    applicable_c_skills,
    build_cases,
    load_config,
    parent_config,
)
from nano_harness.semantic_skill_execution import (
    SKILL_PROMPTS as AB_SKILL_PROMPTS,
)
from nano_harness.verified_tool_execution import (
    contamination_audit,
    public_case_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/qwen35_router_skill_registry_v5.json"
JSON_OUTPUT = (
    ROOT / "docs/experiments/qwen35_router_skill_registry_v5.preregister.json"
)
MARKDOWN_OUTPUT = ROOT / "docs/experiments/qwen35_router_skill_registry_v5.md"
PRIOR_CONFIGS = {
    "integration_v1": (
        ROOT / "configs/harness/qwen35_router_adapter_integration_v1.json",
        load_v1,
        build_v1,
    ),
    "integration_v2": (
        ROOT / "configs/harness/qwen35_router_adapter_integration_v2.json",
        load_v2,
        build_v2,
    ),
    "integration_v3": (
        ROOT / "configs/harness/qwen35_router_adapter_integration_v3.json",
        load_v3,
        build_v3,
    ),
    "skill_fallback_v4": (
        ROOT / "configs/harness/qwen35_router_skill_fallback_v4.json",
        load_v4,
        build_v4,
    ),
}


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(" ".join(prompt.casefold().split()).encode()).hexdigest()


def training_prompts(path: Path) -> list[str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return [
        next(message["content"] for message in row["messages"] if message["role"] == "user")
        for row in document["samples"]
    ]


def build_receipt() -> dict:
    config = load_config(CONFIG)
    _, parent = parent_config(config)
    cases = build_cases(config)
    families = (*POSITIVE_FAMILIES, *C_FAMILIES)
    family_counts = Counter(case["family"] for case in cases)
    label_counts = Counter(case["expected_label"] for case in cases)
    if family_counts != Counter({family: 16 for family in families}):
        raise ValueError("V5 family balance differs")
    if label_counts != Counter({"A": 16, "B": 16, "C": 128}):
        raise ValueError("V5 label balance differs")
    registry = {
        case["case_id"]: applicable_c_skills(case["prompt"]) for case in cases
    }
    if any(
        registry[case["case_id"]]
        != ([case["family"]] if case["family"] in C_FAMILIES else [])
        for case in cases
    ):
        raise ValueError("V5 registry applicability differs")

    train = training_prompts(Path(config.router_training_data_path))
    fresh = {prompt_hash(case["prompt"]) for case in cases}
    prior_cases = {
        name: builder(loader(path))
        for name, (path, loader, builder) in PRIOR_CONFIGS.items()
    }
    overlap = {
        "router_training": len(fresh & {prompt_hash(prompt) for prompt in train}),
        **{
            name: len(fresh & {prompt_hash(case["prompt"]) for case in rows})
            for name, rows in prior_cases.items()
        },
    }
    audit = contamination_audit(parent, cases)
    if (
        any(overlap.values())
        or any(audit["prior_surface_prompt_overlap"].values())
        or any(audit["benchmark_prompt_overlap"].values())
    ):
        raise ValueError("V5 surface is not history-disjoint")

    tokenizer = AutoTokenizer.from_pretrained(
        parent.four_b_model_path, local_files_only=True
    )
    route_max = max(
        len(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": ROUTER_SYSTEM},
                        {"role": "user", "content": case["prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    **parent.chat_template_kwargs,
                )
            )
        )
        for case in cases
    )
    skill_max = {
        family: max(
            len(
                tokenizer.encode(
                    tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": (
                                    AB_SKILL_PROMPTS[family]
                                    if family in POSITIVE_FAMILIES
                                    else SKILL_PROMPTS[family]
                                ),
                            },
                            {"role": "user", "content": case["prompt"]},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        **parent.chat_template_kwargs,
                    )
                )
            )
            for case in cases
            if case["family"] == family
        )
        for family in families
    }
    if (
        route_max + config.route_max_tokens > parent.max_model_len
        or max(skill_max.values()) + config.plan_max_tokens > parent.max_model_len
    ):
        raise ValueError("V5 context budget differs")

    contract = public_case_contract(cases)
    return {
        "schema_version": "nano_harness_router_skill_registry_preregister_v5",
        "experiment_id": config.experiment_id,
        "identity": {
            "code_revision": git_revision(),
            "config_sha256": sha256_file(CONFIG),
            "case_contract_sha256": contract["case_contract_sha256"],
            "v4_report_sha256": config.v4_report_sha256,
            "router_training_data_sha256": config.router_training_data_sha256,
            "adapter_sha256": config.adapter_tree_sha256,
        },
        "case_contract": contract,
        "freshness": {
            "case_seed": config.case_seed,
            "value_offset": config.value_offset,
            "prompt_overlap": overlap,
            "prior_surface_prompt_overlap": audit["prior_surface_prompt_overlap"],
            "benchmark_prompt_overlap": audit["benchmark_prompt_overlap"],
            "benchmark_rows_hashed": audit["benchmark_rows_hashed"],
            "training_prompts_hashed": len(train),
            "prior_outputs_loaded": False,
            "benchmark_outputs_loaded": False,
            "canary_or_holdout_loaded": False,
        },
        "surface": {
            "cases": 160,
            "family_counts": dict(sorted(family_counts.items())),
            "label_counts": dict(sorted(label_counts.items())),
        },
        "registry": {
            "policy": config.skill_registry_policy,
            "c_unique_matches": sum(bool(value) for value in registry.values()),
            "ab_false_matches": sum(
                bool(registry[case["case_id"]])
                for case in cases
                if case["family"] in POSITIVE_FAMILIES
            ),
            "skill_prompt_sha256": {
                family: hashlib.sha256(SKILL_PROMPTS[family].encode()).hexdigest()
                for family in C_FAMILIES
            },
            "skill_regex": SKILL_REGEX,
            "maximum_input_tokens": skill_max,
            "uses_case_metadata": False,
            "uses_expected_answer": False,
            "uses_case_correctness": False,
        },
        "acceptance": {
            "router_correct_160": True,
            "registry_unique_c_128_ab_false_zero": True,
            "ab_verified_32": True,
            "c_single_skill_verified_128": True,
            "fallbacks_zero": True,
            "candidate_vs_four_significant_zero_loss": True,
            "candidate_vs_nine_significant_zero_loss": True,
            "every_family_non_regression": True,
            "benchmark_treatment_preregistration_allowed_after_pass": True,
            "benchmark_generation_allowed_after_pass": False,
            "v1_v2_v3_v4_v5_rerun_allowed_after_pass": False,
        },
        "decision_policy": {
            "passed": "Publish V5 and preregister benchmark treatment.",
            "failed": "Publish V5 negative evidence; do not rerun or tune V1-V5.",
        },
        "execution_boundary": config.execution_boundary,
        "claim_boundary": (
            "This preregisters a target-blind applicability registry plus "
            "single-schema skill extraction on a fresh synthetic surface."
        ),
    }


def render_markdown(receipt: dict) -> str:
    return f"""# Qwen3.5 Router Skill Registry v5

V4 improved the candidate to 142/160 but its shared eight-schema selector
failed on 20 cases. V5 uses target-blind applicability predicates and exposes
exactly one schema after a unique match.

- fresh cases: 160, ten families x 16;
- C registry unique matches: 128/128;
- A/B false registry matches: 0/32;
- overlap with training, V1-V4, prior surfaces, GSM8K, MMLU, GPQA: zero;
- config SHA: `{receipt['identity']['config_sha256']}`;
- case contract SHA: `{receipt['identity']['case_contract_sha256']}`;
- model generation started: false.

Passing permits only separately pre-registered benchmark treatment generation.
V1-V5 cannot be rerun after observation.
"""


def main() -> None:
    receipt = build_receipt()
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    MARKDOWN_OUTPUT.write_text(render_markdown(receipt), encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": receipt["experiment_id"],
                "identity": receipt["identity"],
                "freshness": receipt["freshness"],
                "registry": receipt["registry"],
                "acceptance": receipt["acceptance"],
                "execution_boundary": receipt["execution_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
