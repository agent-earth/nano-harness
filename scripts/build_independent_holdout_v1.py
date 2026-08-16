#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASET_ROOT = ROOT / "../../datasets"
MANIFEST = ROOT / "configs/harness/qwen35_independent_holdout_v1.yaml"
SELECTION = (
    ROOT / "configs/generated/qwen35_independent_holdout_v1_selection.json"
)
SELECTION_SEED = "qwen35-independent-holdout-v1-unseen-source-indices"
EXCLUDED_RESULT_PREFIX = "qwen35-independent-holdout-v1"
SPECS = {
    "gsm8k": {
        "path": "gsm8k/gsm8k/main/test-00000-of-00001.parquet",
        "sha256": (
            "ee7b8da9e381df27b9e3f7758a159ab"
            "2bdaa4dbaa910546cbbc47e0cb44e4f59"
        ),
        "scorer": "numeric_exact",
        "limit": 16,
        "max_tokens": 600,
    },
    "mmlu": {
        "path": (
            "mmlu_no_train/mmlu_no_train/all/"
            "test-00000-of-00001.parquet"
        ),
        "sha256": (
            "02033371a64dbe5a0d8b6fb9d612900"
            "afcd0fea5140e53490993a4540b3a58fd"
        ),
        "scorer": "choice_exact",
        "limit": 16,
        "max_tokens": 32,
        "answer_only": True,
        "system_prompt": (
            "Solve the problem internally. Return only the requested FINAL "
            "line and do not use tools or external information."
        ),
    },
    "gpqa_diamond": {
        "path": (
            "GPQA-Diamond/GPQA-Diamond/test/"
            "gpqa_diamond.parquet"
        ),
        "sha256": (
            "fdd6e95117cdf87075f56bf673a5bae4"
            "680b143bc2d29b486470810122c33f39"
        ),
        "scorer": "choice_exact",
        "limit": 8,
        "max_tokens": 32,
        "max_source_chars": 1200,
        "answer_only": True,
        "system_prompt": (
            "Solve the problem internally. Return only the requested FINAL "
            "line and do not use tools or external information."
        ),
    },
}


def canonical_sha256(value: object) -> str:
    rendered = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def history_snapshot() -> tuple[dict[str, set[int]], list[dict[str, object]]]:
    seen: dict[str, set[int]] = defaultdict(set)
    files = []
    for path in sorted(RESULTS.rglob("*.jsonl")):
        relative = path.relative_to(RESULTS).as_posix()
        if EXCLUDED_RESULT_PREFIX in relative:
            continue
        rows = 0
        benchmark_rows = 0
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            benchmark = row.get("benchmark")
            source_index = row.get("source_index")
            if benchmark in SPECS and isinstance(source_index, int):
                seen[str(benchmark)].add(source_index)
                benchmark_rows += 1
        if benchmark_rows:
            files.append(
                {
                    "path_sha256": hashlib.sha256(
                        relative.encode("utf-8")
                    ).hexdigest(),
                    "content_sha256": sha256_file(path),
                    "rows": rows,
                    "benchmark_rows": benchmark_rows,
                }
            )
    return seen, files


def eligible_indices(benchmark: str, spec: dict[str, object]) -> list[int]:
    path = (DATASET_ROOT / str(spec["path"])).resolve()
    if sha256_file(path) != spec["sha256"]:
        raise SystemExit(f"{benchmark} dataset SHA256 mismatch")
    metadata = pq.read_metadata(path)
    indices = list(range(metadata.num_rows))
    max_chars = spec.get("max_source_chars")
    if max_chars is not None:
        questions = pq.read_table(path, columns=["question"])["question"]
        indices = [
            index
            for index, value in enumerate(questions)
            if len(str(value.as_py()).strip()) <= int(max_chars)
        ]
    return indices


def select(
    benchmark: str,
    eligible: list[int],
    seen: set[int],
    limit: int,
) -> list[int]:
    candidates = [index for index in eligible if index not in seen]
    candidates.sort(
        key=lambda index: hashlib.sha256(
            f"{SELECTION_SEED}\0{benchmark}\0{index}".encode("utf-8")
        ).hexdigest()
    )
    if len(candidates) < limit:
        raise SystemExit(
            f"{benchmark} has only {len(candidates)} unseen eligible rows"
        )
    return candidates[:limit]


def main() -> None:
    seen, files = history_snapshot()
    selection = {}
    dataset_receipts = {}
    for benchmark, spec in SPECS.items():
        eligible = eligible_indices(benchmark, spec)
        selected = select(
            benchmark,
            eligible,
            seen[benchmark],
            int(spec["limit"]),
        )
        if set(selected) & seen[benchmark]:
            raise SystemExit(f"{benchmark} holdout overlaps history")
        selection[benchmark] = selected
        dataset_receipts[benchmark] = {
            "dataset_sha256": spec["sha256"],
            "dataset_rows": pq.read_metadata(
                (DATASET_ROOT / str(spec["path"])).resolve()
            ).num_rows,
            "eligible_rows": len(eligible),
            "historically_seen_rows": len(seen[benchmark]),
            "historically_seen_indices_sha256": canonical_sha256(
                sorted(seen[benchmark])
            ),
            "selected_rows": len(selected),
            "selected_indices": selected,
            "selected_indices_sha256": canonical_sha256(selected),
            "history_overlap": 0,
        }

    manifest = {
        "schema_version": "nano_harness_baseline_suite_v1",
        "suite_id": "qwen35-independent-holdout-v1",
        "selection_seed": SELECTION_SEED,
        "strategy": "direct",
        "system_prompt": (
            "You are being evaluated under a deterministic answer contract. "
            "Solve the problem yourself, follow the requested FINAL line "
            "exactly, and do not use tools or external information."
        ),
        "max_tokens": 600,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "datasets": [
            {
                "name": benchmark,
                **{
                    key: value
                    for key, value in spec.items()
                    if key
                    in {
                        "path",
                        "sha256",
                        "scorer",
                        "limit",
                        "max_tokens",
                        "max_source_chars",
                        "answer_only",
                        "system_prompt",
                    }
                },
                "indices": selection[benchmark],
            }
            for benchmark, spec in SPECS.items()
        ],
    }
    receipt = {
        "schema_version": "nano_harness_unseen_holdout_selection_v1",
        "suite_id": manifest["suite_id"],
        "selection_seed": SELECTION_SEED,
        "policy": {
            "selection_uses_source_indices_only": True,
            "selection_uses_question_length_for_eligibility": True,
            "selection_uses_question_content": False,
            "selection_uses_references": False,
            "selection_uses_model_outputs": False,
            "selection_uses_scores": False,
            "prompts_loaded_before_evaluation": False,
            "case_manifest_generated_before_evaluation": False,
            "training_eligible": False,
            "purpose": "post_intervention_independent_holdout",
        },
        "history_snapshot": {
            "result_files": len(files),
            "result_files_digest": canonical_sha256(files),
            "excluded_result_prefix": EXCLUDED_RESULT_PREFIX,
        },
        "datasets": dataset_receipts,
        "summary": {
            "cases": sum(len(indices) for indices in selection.values()),
            "by_benchmark": {
                benchmark: len(indices)
                for benchmark, indices in selection.items()
            },
            "history_overlap": 0,
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "suite_id": manifest["suite_id"],
                "cases": receipt["summary"]["cases"],
                "by_benchmark": receipt["summary"]["by_benchmark"],
                "history_overlap": 0,
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "selection_receipt": str(SELECTION.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
