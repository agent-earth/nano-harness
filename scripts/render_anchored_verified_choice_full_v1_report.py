#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from nano_harness.baseline import compare_baselines, load_cases, load_manifest
from nano_harness.verified_choice import verify_explicit_average_choice


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/harness/anchored_v1_verified_choice_full_v1.json"
RESULT = (
    ROOT
    / "results/harness/anchored-v1-verified-choice-full-v1/"
    "candidate/result.json"
)
PRE_REGISTRATION_REVISION = "33fa95b"
CONFIG_SHA256 = (
    "27202f9c43325366c8cf35070e9d040b"
    "8f5ee52ce8745f1e15b6f3f7d732c1dd"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise SystemExit("verified choice full config differs")
    raw = json.loads(RESULT.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "nano_harness_verified_choice_full_result_v1":
        raise SystemExit("unexpected verified choice full schema")
    boundary = raw["evaluation_boundary"]
    if boundary != {
        "target_used_by_parser": False,
        "sealed_canary_passed": True,
        "prior_full_suite_run": True,
        "independent_holdout_run": False,
    }:
        raise SystemExit("verified choice full boundary differs")

    config = raw["config"]
    manifest = load_manifest(config["manifest_path"])
    cases = load_cases(manifest, Path(config["dataset_root"]))
    by_case = {case.case_id: case for case in cases}
    baseline_rows = {row["case_id"]: row for row in raw["baseline_rows"]}
    candidate_rows = {row["case_id"]: row for row in raw["candidate_rows"]}
    if (
        len(cases) != 211
        or set(by_case) != set(baseline_rows)
        or set(by_case) != set(candidate_rows)
    ):
        raise SystemExit("verified choice full row identities differ")

    choice_ids = {
        case.case_id for case in cases if case.scorer == "choice_exact"
    }
    if set(raw["receipts"]) != choice_ids or len(choice_ids) != 115:
        raise SystemExit("verified choice full receipt coverage differs")
    for case_id in sorted(choice_ids):
        if (
            verify_explicit_average_choice(by_case[case_id].prompt)
            != raw["receipts"][case_id]
        ):
            raise SystemExit("verified choice full receipt is not reproducible")

    changed = [
        case_id
        for case_id in by_case
        if baseline_rows[case_id]["output"]
        != candidate_rows[case_id]["output"]
    ]
    regressions = [
        case_id
        for case_id in by_case
        if baseline_rows[case_id]["score"]
        and not candidate_rows[case_id]["score"]
    ]
    reasons = Counter(
        receipt["reason"] for receipt in raw["receipts"].values()
    )
    if (
        changed
        or regressions
        or raw["routing"]["verified_overrides"] != 0
        or raw["routing"]["fallback_rows"] != 211
        or reasons != Counter(
            {"unsupported_intent": 114, "expression_count_not_two": 1}
        )
    ):
        raise SystemExit("verified choice full fallback evidence differs")

    candidate_list = list(candidate_rows.values())
    with tempfile.TemporaryDirectory() as directory:
        candidate_path = Path(directory) / "candidate.jsonl"
        write_jsonl(candidate_path, candidate_list)
        versus_four = compare_baselines(
            candidate_path,
            Path(config["four_b_raw_path"]),
        )
        versus_nine = compare_baselines(
            candidate_path,
            Path(config["nine_b_raw_path"]),
        )

    baseline = raw["baseline"]
    candidate = raw["candidate"]
    task_thresholds = {
        "gsm8k": 90,
        "mmlu": 67,
        "gpqa_diamond": 6,
    }
    checks = {
        "baseline_total_164": baseline["correct"] == 164,
        "candidate_total_at_least_163": candidate["correct"] >= 163,
        "gsm8k_at_least_base_90": (
            candidate["by_benchmark"]["gsm8k"]["correct"] >= 90
        ),
        "mmlu_at_least_base_67": (
            candidate["by_benchmark"]["mmlu"]["correct"] >= 67
        ),
        "gpqa_at_least_base_6": (
            candidate["by_benchmark"]["gpqa_diamond"]["correct"] >= 6
        ),
        "macro_at_least_base": (
            versus_four["candidate_macro_accuracy"]
            >= versus_four["baseline_macro_accuracy"]
        ),
        "micro_not_significantly_negative": (
            versus_four["overall_micro"]["paired_bootstrap_95_ci"][1] >= 0
        ),
        "zero_output_changes": not changed,
        "zero_regressions": not regressions,
        "all_fallback_outputs_equal_direct": (
            raw["routing"]["fallback_rows"] == 211
        ),
        "target_blind_parser": boundary["target_used_by_parser"] is False,
    }
    if [key for key, value in checks.items() if not value] != [
        "mmlu_at_least_base_67"
    ]:
        raise SystemExit(
            "verified choice full failed outside expected MMLU gate: "
            + ",".join(key for key, value in checks.items() if not value)
        )

    nine_micro = versus_nine["overall_micro"]
    significantly_exceeds_nine = (
        versus_nine["candidate_macro_accuracy"]
        > versus_nine["baseline_macro_accuracy"]
        and nine_micro["paired_bootstrap_95_ci"][0] > 0
        and nine_micro["mcnemar_exact_p"] < 0.05
        and all(
            metric["delta"] >= 0
            for metric in versus_nine["benchmarks"].values()
        )
    )
    report = {
        "schema_version": "nano_harness_public_anchored_verified_choice_full_v1",
        "experiment_id": raw["experiment_id"],
        "pre_registration_revision": PRE_REGISTRATION_REVISION,
        "passed": False,
        "identity": {
            "config_sha256": CONFIG_SHA256,
            "raw_result_sha256": sha256_file(RESULT),
            **raw["identity"],
        },
        "baseline": baseline,
        "candidate": candidate,
        "routing": {
            **raw["routing"],
            "reason_counts": dict(sorted(reasons.items())),
            "changed_outputs": len(changed),
            "regressions": len(regressions),
        },
        "comparisons": {
            "candidate_vs_four_b": versus_four,
            "candidate_vs_nine_b": versus_nine,
        },
        "full_gate": checks,
        "evaluation_boundary": {
            **boundary,
            "independent_holdout_prompts_loaded": False,
            "independent_holdout_references_loaded": False,
        },
        "decision": {
            "passed_four_b_non_regression": False,
            "significantly_exceeds_nine_b": significantly_exceeds_nine,
            "independent_holdout_allowed": False,
            "merge_allowed": False,
            "scale_allowed": False,
            "rl_allowed": False,
            "post_hoc_parser_expansion_allowed": False,
            "next_action": (
                "Preserve anchored-v1 plus the narrow verified executor as a "
                "local capability signal, but do not open the independent "
                "holdout. The parser makes zero old-suite overrides and MMLU "
                "remains 66 versus base 67. Replan from fresh generic data, "
                "not observed benchmark prompts."
            ),
        },
    }
    four_micro = versus_four["overall_micro"]
    markdown = f"""# Anchored-v1 Verified Choice Full Development v1 Result

## Result

The executor preserves all old-suite outputs but does not transfer:

| Benchmark | Candidate | Base 4B | 9B |
| --- | ---: | ---: | ---: |
| GSM8K | {candidate['by_benchmark']['gsm8k']['correct']}/96 | 90/96 | 89/96 |
| MMLU | {candidate['by_benchmark']['mmlu']['correct']}/96 | 67/96 | 58/96 |
| GPQA-Diamond | {candidate['by_benchmark']['gpqa_diamond']['correct']}/19 | 6/19 | 4/19 |

Candidate remains {candidate['correct']}/211. All 115 choice prompts are
outside parser v1's narrow explicit arithmetic-average contract, so there are
zero overrides, 211/211 fallback parity, and zero regressions.

Versus base 4B, micro delta is {four_micro['delta']:+.4f}, 95% CI
[{four_micro['paired_bootstrap_95_ci'][0]:+.4f},
{four_micro['paired_bootstrap_95_ci'][1]:+.4f}], and McNemar
p={four_micro['mcnemar_exact_p']:.3f}. MMLU remains one case below base, so
the frozen per-task non-regression gate fails.

Versus 9B, micro delta is {nine_micro['delta']:+.4f}, 95% CI
[{nine_micro['paired_bootstrap_95_ci'][0]:+.4f},
{nine_micro['paired_bootstrap_95_ci'][1]:+.4f}], and McNemar
p={nine_micro['mcnemar_exact_p']:.3f}. The dual significance criterion still
does not pass.

## Decision

Do not open the independent holdout. Do not expand parser v1 from observed
benchmark prompts. Preserve the local verified-execution signal and replan
from fresh generic data. Merge, scale, and RL remain forbidden.

## Identity

- pre-registration revision: `{PRE_REGISTRATION_REVISION}`;
- config SHA256: `{CONFIG_SHA256}`;
- suite manifest SHA256: `{raw['identity']['manifest']}`;
- anchored-v1 raw SHA256: `{raw['identity']['baseline_raw']}`;
- canary pass SHA256: `{raw['identity']['canary_pass_report']}`;
- raw applicator result SHA256: `{sha256_file(RESULT)}`.
"""
    output = ROOT / "docs/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "anchored_v1_verified_choice_full_v1.public.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "anchored_v1_verified_choice_full_v1.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": False,
                "correct": candidate["correct"],
                "failed_gate": "mmlu_at_least_base_67",
                "independent_holdout_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
