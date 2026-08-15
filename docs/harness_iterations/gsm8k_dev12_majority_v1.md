# GSM8K Dev12 Deterministic Majority Protocol

## Hypothesis

Remove learned arbitration. Run:

1. protected 600-token direct;
2. independent 384-token forward solve;
3. independent 384-token verification-first solve.

Normalize all numeric predictions. If any value appears at least twice, select
that majority. If there is no majority, keep protected direct. Code emits the
final numeric contract deterministically.

The two re-solves use different system and user instructions and never see
direct or each other's output.

## Fresh Slice

GSM8K dev12 uses `start: 408, limit: 24`, the first continuous 24-case window
with unique IDs and zero overlap against all historical/sealed manifests.

Three matched arms:

1. Qwen3.5-4B direct;
2. Qwen3.5-4B protected three-way numeric majority;
3. Qwen3.5-9B direct.

## Directional Decision

Promote only if treatment:

- has higher point accuracy than 4B direct;
- has at least one treatment-only win and zero direct-only losses;
- has no API or final parse errors;
- has protected-direct predictions identical to direct;
- passes case, strategy, solve-independence, and deterministic-vote audits.

Report majority count, no-majority count, wins, losses, truncations, tokens,
and wall time. Do not tune on dev12.

## Result

Deterministic majority, 4B direct, and 9B direct each score 23/24. Majority
changes zero of 24 direct predictions: 16 cases have a majority that includes
direct, and 8 have no majority and keep direct.

The treatment uses 33,493 tokens and 865.8s, with 48 re-solve calls and 32
re-solve truncations, but creates no repair. Dev12 fails.

Stop full-suite re-solving. Fresh dev13 should run a single recovery solve only
when protected direct is unparseable; parseable direct outputs are returned
unchanged by construction.

- [`docs/results/gsm8k_dev12_majority_v1.md`](../results/gsm8k_dev12_majority_v1.md)
- [`docs/results/gsm8k_dev12_majority_v1.public.json`](../results/gsm8k_dev12_majority_v1.public.json)
