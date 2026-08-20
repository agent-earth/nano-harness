# Qwen3.5 Router Adapter Integration v1

## Purpose

The exact router adapter `48d72666cf269f64825791801c71aad5ae1d9e97cbc70574c216b12974e46e63` is frozen
before serving. It only emits `FINAL: A/B/C`; the unchanged base 4B performs
typed planning and verified feedback. `C` preserves the base direct answer.

## Fresh Surface

- 128 cases: 32 per family;
- positive A/B: 64; unsupported C: 64;
- exact normalized prompt overlap with 960 SFT rows, 256 prior multiclass
  rows, and 128 prior binary rows: all zero;
- benchmark, canary, and holdout rows or outputs loaded: false.

## Gates

- A 32/32, B 32/32, C 64/64, zero unsupported false routes;
- 64 verified executions, 64 feedback matches, zero fallbacks;
- unsupported candidate exactly preserves direct scoring fields;
- candidate significantly beats both direct 4B and direct 9B with at least
  12 wins and zero losses;
- every family is non-regressing.

Passing permits only a separately pre-registered question-only scan. It does
not permit benchmark generation, canary rerun, holdout access, training, or RL.

## Frozen Identity

- config SHA: `4eb7000201530ecb2ced96f4b1d490d115f4c1e1c6a6cb008cf64d5dc403d4c4`;
- case contract SHA: `d14d2e641c7af4dd11b2522b9ef237c8b89d8ccdbb2076971da292bc4aed49fe`;
- SFT report SHA: `c8af17cfa2fb77b594a9b34deaeccf27273491da6c350c3c5deb1435a9336c69`;
- adapter SHA: `48d72666cf269f64825791801c71aad5ae1d9e97cbc70574c216b12974e46e63`;
- model generation started: false;
- adapter service started: false.
