# Draft-Verify v1

## Evidence Integrity Correction

Revision `cf2a00e` actually sent `case.prompt` to the treatment draft. For
answer-only MMLU and GPQA, this was not the reasoning-oriented
`case.draft_prompt` modeled by validation. Fixed-v5 scores remain observations
of that implementation but do not validate the stated reasoning-draft
mechanism.

The dev direct arm had the inverse mismatch: it received `case.draft_prompt`
under the 32-token answer-only budget. Its MMLU and GPQA controls are invalid,
so the dev promotion claim is withdrawn. Raw evidence is retained for audit.

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

## Result

The disjoint dev slice observed 5/18 versus 9/18, but the comparison is
invalid because the direct MMLU and GPQA controls used a mismatched
prompt/budget contract. It cannot establish promotion.

On fixed v5 evaluation, 4B draft-verify reached 52/72 (macro 0.7222), compared
with 4B direct at 50/72 and 9B direct at 51/72. The 4B-vs-9B paired 95%
bootstrap interval still crosses zero, and GPQA regressed from 10/24 to 8/24.
The observed answer-only-draft strategy does not satisfy harness-stage
acceptance. A corrected fresh experiment is required before claiming the
reasoning-draft mechanism.

See the public-safe report:

- [`docs/results/draft_verify_v1.md`](../results/draft_verify_v1.md)
- [`docs/results/draft_verify_v1.public.json`](../results/draft_verify_v1.public.json)
