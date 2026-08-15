# Draft-Verify v1

## Hypothesis

A small model may need room to reason, but the final answer contract should not
compete with that reasoning for the same completion. A two-stage harness may
improve correctness and format reliability:

1. a bounded draft pass produces compact analysis and a candidate;
2. an independent verifier sees the original task and draft, corrects the
   candidate when needed, and emits only the required `FINAL:` line.

The treatment changes harness policy only. Model weights, selected cases,
dataset revisions, decoding temperature, chat-template thinking setting, and
scorers remain fixed.

## Development Slice

The development slice uses `start: 24` and `limit: 6` for GSM8K, MMLU, and
GPQA-Diamond after the same seeded hash ordering as the v5 baseline. Its 18
case IDs have zero overlap with the fixed 72-case evaluation suite.

- Control: `configs/harness/qwen35_dev_direct_v1.yaml`
- Treatment: `configs/harness/qwen35_dev_draft_verify_v1.yaml`
- Draft budget: 384 tokens
- Verifier budget: 32 tokens

The committed control and treatment case manifests have the same SHA256 and
contain no task bodies.

## Evidence Contract

Each treatment result records:

- final score, normalized prediction, and finish reason;
- combined prompt/completion tokens and wall-clock latency;
- draft and verifier token budgets, usage, and finish reasons;
- SHA256 of the draft output rather than a public copy of its text.

Raw outputs remain local and ignored.

## Decision Rule

Promote to the fixed v5 evaluation only if the treatment has:

- positive paired macro and micro deltas on the 18-case dev slice;
- no model API errors;
- no new final-answer parse failures;
- no material regression isolated to one task group without a clear mechanism;
- a cost increase justified by additional correct cases.

If the treatment is neutral or negative, retain the evidence, reject the
strategy, and test a narrower mechanism rather than tuning on v5 evaluation
cases.
