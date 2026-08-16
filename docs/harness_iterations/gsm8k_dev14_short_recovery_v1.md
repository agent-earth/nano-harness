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
