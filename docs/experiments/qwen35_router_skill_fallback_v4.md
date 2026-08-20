# Qwen3.5 Router Skill Fallback v4

## Purpose

V3 proved the router (160/160) and A/B verified execution, but rejected
`C -> 4B direct` because it inherited 14 9B-only wins. V4 changes only the C
policy: a model selects one of eight typed skills, a deterministic verifier
checks copied facts, and the verified integer is emitted without model rewrite.

## Freshness

- seed `20260829`, value offset
  `12000000`;
- A/B = 16/16 and eight C skills = 8 x 16;
- overlap with 7,680 training prompts, V1/V2/V3 prompts, prior surfaces, and
  complete GSM8K/MMLU/GPQA prompt columns: all zero;
- prior outputs, benchmark outputs, canary, and holdout: not loaded.

## Gates

- router 160/160;
- A/B verified executions 32/32;
- C typed-skill verified executions and exact results 128/128;
- zero fallbacks;
- significant zero-loss superiority over direct 4B and 9B;
- every-family non-regression.

Passing permits only a separately pre-registered benchmark treatment. V1-V4
cannot be rerun after observation. Benchmark generation, canary, holdout,
training, and RL remain closed.

## Identity

- config SHA: `240b6e5273c5c0bd111e5c76fa3ab37697f19f6d7666d64129ecdb6d00563f89`;
- case contract SHA: `4ef38ad28852928cd9b584ca509290d7327123fdbb3c442242e16da3b3194b01`;
- V3 report SHA: `cfbd0edbc74739eb1d0a860c19cca2c07edfd52093d7ccf7f86e114f33a2ac03`;
- generation started: false.
