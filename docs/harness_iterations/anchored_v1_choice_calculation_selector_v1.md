# Anchored-v1 Choice Calculation Selector v1

## Hypothesis

Anchored-v1 scores 6/8 on the generic synthetic choice development family.
One fixed choice-replay continuation moves a wrong option to another wrong
option without improving any score. The remaining failures are arithmetic
participant-average problems, so an inference-time calculation and constrained
selector may improve answer mapping without changing model weights.

## Frozen Method

Evaluate the unchanged 32-row development split from generic choice replay
v11:

- first run the exact anchored-v1 model directly on all 32 rows;
- route only the 8 `final_choice` rows through the treatment;
- calculation stage: 128 tokens, explicit intermediate arithmetic and option
  comparison, ending in `CANDIDATE: <letter>`;
- selector stage: 8 tokens, independently verify the calculation, then decode
  under regex `FINAL: [A-D]`;
- reuse the direct baseline output byte-for-byte for all 16 numeric and 8
  process rows.

Temperature is 0 and thinking is disabled. Prompts, budgets, regex, dataset,
adapter-serving receipt, and model name are frozen in the committed config.
No alternate prompt, stage budget, regex, retry policy, or route will be tried
after observing the result.

## Evidence Boundary

The method uses only deterministic synthetic development rows. It does not load
benchmark, canary, independent-holdout, teacher, model-feedback, or training
payloads. The development split provides only a local method gate.

The old canary may run only if the local gate passes. The old 211-case suite
may run only if that canary also passes. The independent holdout remains unread
until old-suite per-task base non-regression passes.

## Frozen Identity

- dataset SHA256:
  `4657e96af9f9d1b81bfdb5fac6a29c31baf24b23d7753508aafdea5603ffd80d`;
- anchored serving receipt SHA256:
  `2549527942acfe53a1eb352453649a9ea3cc31d68bb9790c865553ee95c2f578`;
- serving adapter weights SHA256:
  `9ce7be3954f8e0f3d245fe846d6e35275243b7f0caf66cb847fd716173658649`.
- config SHA256:
  `8727fb9f3d97c860f551fac2046822feec64898e9fe5005271f64df8e247674a`.

## Local Gate

The direct baseline must exactly reproduce anchored-v1:

- strict 22/32 and semantic 25/32;
- numeric 11/16, choice 6/8, process 8/8 semantic.

The candidate must satisfy all of:

- strict >=22/32 and semantic >=25/32;
- numeric >=11/16 and process =8/8;
- choice >=7/8;
- 24/24 non-choice outputs exactly equal the direct baseline;
- 8/8 selectors match the frozen regex;
- no API errors or missing rows.

Failure ends this exact method and blocks canary, full development, holdout,
merge, scale, and RL.

## Reproduction

```bash
PYTHONPATH=.:../nano-train NANO_HARNESS_API_KEY=local-vllm \
  ../.venv/bin/python -m nano_harness.cli analog-contract \
  --config \
    configs/harness/anchored_v1_choice_calculation_selector_v1.json
```
