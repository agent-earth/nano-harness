# Complete Conditional-Majority v1 Execution Addendum

This addendum changes execution scheduling only. It does not change the
pre-registered model, case set, prompts, parser, sampling parameters, global
case-index seeds, vote thresholds, scorer, frozen MMLU endpoint, or frozen
GPQA endpoint.

- The first 156 globally sorted GSM8K cases were complete before this
  addendum.
- Their immutable merged prefix SHA256 is
  `0ea77ce8ea4c2ec7a022939afcc5bdd6ffbea18a83f29c8ca4229dc6e0067d92`.
- The remaining 1,163 cases are partitioned by
  `global_sorted_case_index mod 8`.
- Shards 0/2/4/6 use one Qwen3.5-4B service; shards 1/3/5/7 use a second
  byte-identical Qwen3.5-4B service.
- Both services use the same model config SHA256 and 2,048-token context.
  Only server batch capacity changes from one to four concurrent sequences.
- Merge requires the prefix and all eight shards to be pairwise disjoint and
  to cover exactly all 1,319 frozen GSM8K case IDs.

The addendum was written after observing only progress and aggregate route
counts for the prefix. No prompt, answer, per-case correctness, or replacement
policy was changed from the parent pre-registration. Prefix cases will not be
regenerated. No further scheduling or policy change is allowed after this
addendum.
