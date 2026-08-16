# GSM8K Dev13 Conditional Recovery Protocol

## Hypothesis

Run the matched 600-token direct path first.

- If its numeric `FINAL:` parses, return the same normalized number without
  another model call.
- Only if direct is unparseable, run one 384-token recovery solver and return
  its parsed number deterministically.

This structurally preserves every parseable direct answer and concentrates
extra inference on observable format failures.

## Fresh Slice

GSM8K dev13 uses `start: 432, limit: 48`, the first continuous 48-case window
with unique IDs and zero overlap against all historical/sealed manifests. The
larger directional sample increases the chance of observing direct parse
failures without selecting cases by labels or outputs.

Three matched arms:

1. Qwen3.5-4B direct;
2. Qwen3.5-4B conditional recovery;
3. Qwen3.5-9B direct.

## Directional Decision

Promote only if:

- recovery triggers at least once and produces at least one treatment-only win;
- there are zero direct-only losses;
- treatment parse failures do not exceed direct parse failures;
- no API errors occur;
- protected-direct predictions match direct and all conditional-call/selection
  audits pass;
- average treatment token cost is below 1.5x direct.

Do not tune on dev13.

## Result

Conditional recovery and 4B direct both score 39/48; 9B direct scores 44/48.
Recovery triggers exactly on the four direct parse failures, but all four
384-token recovery calls also truncate without a numeric final. Parseable
direct outputs are unchanged and token ratio is 1.127x.

Dev13 fails. Fresh dev14 keeps conditional triggering but replaces long
derivation recovery with a 64-token answer-only call that solves internally
and emits only `FINAL: <number>`.

- [`docs/results/gsm8k_dev13_recovery_v1.md`](../results/gsm8k_dev13_recovery_v1.md)
- [`docs/results/gsm8k_dev13_recovery_v1.public.json`](../results/gsm8k_dev13_recovery_v1.public.json)
