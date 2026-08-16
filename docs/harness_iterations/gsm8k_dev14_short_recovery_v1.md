# GSM8K Dev14 Short Recovery Protocol

## Frozen Change

Keep dev13 conditional triggering and 600-token direct unchanged. When direct
has no parseable numeric final, run one 64-token answer-only recovery:

- solve internally from scratch;
- output only `FINAL: <number>`;
- deterministic code returns its parsed number.

Parseable direct outputs do not invoke recovery and cannot change.

## Fresh Slice

GSM8K dev14 uses `start: 480, limit: 48`, the first continuous 48-case window
with unique IDs and zero overlap against all historical/sealed manifests.

Three matched arms:

1. Qwen3.5-4B direct;
2. Qwen3.5-4B conditional short recovery;
3. Qwen3.5-9B direct.

## Directional Decision

Promote only if:

- recovery triggers and creates at least one treatment-only win;
- there are zero direct-only losses;
- treatment parse failures are fewer than direct parse failures;
- no API errors occur;
- conditional-call/parity/selection audits pass;
- token ratio versus direct is below 1.2x.

Do not tune on dev14.

## Result

Short recovery, 4B direct, and 9B direct each score 45/48. Recovery triggers
once and leaves that case unparseable; parseable direct outputs are unchanged.
Token ratio is 1.014x.

The recovery output calculates the correct four cups per row but spends all 64
tokens explaining the calculation and truncates before `FINAL:`. Prompt-only
answer suppression is insufficient.

Dev14 fails. Fresh dev15 should keep conditional triggering and enforce a
numeric `FINAL:` grammar through the serving API. Do not rescore dev14.

- [`docs/results/gsm8k_dev14_short_recovery_v1.md`](../results/gsm8k_dev14_short_recovery_v1.md)
- [`docs/results/gsm8k_dev14_short_recovery_v1.public.json`](../results/gsm8k_dev14_short_recovery_v1.public.json)
