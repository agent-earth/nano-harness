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
