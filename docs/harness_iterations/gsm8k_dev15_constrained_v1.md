# GSM8K Dev15 Constrained Recovery Protocol

## Capability Evidence

Local vLLM 0.19.1 accepts top-level
`structured_outputs.regex`. A real GPU2 smoke request with regex
`FINAL: [-+]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)` returns exactly
`FINAL: 12`.

## Policy

Keep the 600-token direct path. Only when direct has no parseable numeric
final, run a 32-token recovery with the same numeric `FINAL:` regex enforced by
vLLM. Parseable direct outputs do not invoke recovery and cannot change.

## Fresh Slice

GSM8K dev15 uses `start: 528, limit: 48`, the first continuous 48-case window
with unique IDs and zero overlap against all historical/sealed manifests.

Three matched arms:

1. Qwen3.5-4B direct;
2. Qwen3.5-4B conditional constrained recovery;
3. Qwen3.5-9B direct.

## Directional Decision

Promote only if:

- recovery triggers and creates at least one treatment-only win;
- there are zero direct-only losses;
- treatment parse failures are fewer than direct parse failures;
- every recovery output matches the committed numeric regex;
- conditional execution/parity/selection audits pass;
- token ratio versus direct is below 1.2x;
- no API errors occur.

Do not tune on dev15.

## Result

Constrained recovery and 4B direct each score 46/48; 9B direct scores 47/48.
The 4B direct arm has zero parse failures, so conditional recovery never fires
and treatment runs at exactly 1.0x direct tokens.

Dev15 cannot establish recovery benefit. Keep the policy frozen and
pre-register 96 fresh GSM8K cases to observe the rare direct parse-failure
path.

- [`docs/results/gsm8k_dev15_constrained_recovery_v1.md`](../results/gsm8k_dev15_constrained_recovery_v1.md)
- [`docs/results/gsm8k_dev15_constrained_recovery_v1.public.json`](../results/gsm8k_dev15_constrained_recovery_v1.public.json)
