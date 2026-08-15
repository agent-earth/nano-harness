# Conservative Arbiter Holdout5 Protocol

## Frozen Policy

- GSM8K: direct 4B reasoning;
- MMLU: direct 4B answer-only;
- GPQA-Diamond: protected 4B direct, four independent 96-token option
  evaluators, 64-token conservative arbiter, strict bare-letter normalizer.

No prompts, budgets, routing, scorer, or normalizer changes are allowed after
pre-registration.

## Untouched Cases

Holdout5 uses the first continuous 24-case windows with unique IDs and zero
overlap against every committed historical or sealed manifest:

- GSM8K: `start: 312, limit: 24`;
- MMLU: `start: 120, limit: 24`;
- GPQA-Diamond: `start: 163, limit: 24`.

The first identity-valid GPQA window at position 144 contains one case whose
frozen six-stage contract is 1027 tokens, exceeding the 1024 service limit.
Without reading labels or model outputs, a tokenizer-only scan selected
position 163 as the first continuous 24-case window that is unique,
historically disjoint, and context-valid. Its maximum is 892/1024.

Three matched arms:

1. Qwen3.5-4B direct;
2. Qwen3.5-4B benchmark routing with GPQA conservative arbitration;
3. Qwen3.5-9B direct.

## Acceptance

Harness acceptance requires all of:

- routed 4B macro accuracy above 9B direct;
- routed 4B point accuracy at least 9B on every benchmark;
- paired micro bootstrap 95% lower bound above zero;
- exact McNemar p < 0.05;
- no routed API or final parse errors;
- all case, route, protected-direct, evaluator, arbiter, and input-hash audits
  pass.

Cost and latency are reported, not hard gates. A failure triggers replan; do
not tune on holdout5.
