# Qwen3.5 Complete Conditional-Majority v1

This pre-registers one complete three-benchmark candidate. It starts no model
generation.

## Frozen Candidate

- GSM8K, 1,319 cases: run the admitted target-blind recovered parser and
  conditional-majority v4 policy. Five 4B solves are sampled per case. A
  strict-parse failure may be replaced by 3-of-5 agreement; an already strict
  direct answer requires unanimous 5-of-5 agreement. Otherwise preserve the
  frozen recovered 4B direct answer.
- MMLU, 14,042 cases: preserve the frozen 4B direct result.
- GPQA-Diamond, 198 cases: reuse the frozen V5 conservative-consensus result;
  make no new GPQA request.

The complete candidate is frozen before new GSM8K generation. Existing V5
GSM8K outputs are not reused.

## Sequential Inference

This is the second and final complete GSM8K treatment attempt. GSM8K admission
uses Bonferroni `alpha=0.025` over the two complete attempts. The final
three-benchmark claim additionally applies Holm-Bonferroni at familywise
`alpha=0.05` to the three frozen benchmark p-values.

## Identity

- config SHA: `4f8c138166ada6c03edddfd3205d2cf7b3bc8baf86bdab879fe067f38a2e5013`;
- complete case IDs SHA:
  `d38ee8c3eabbefaf7381253f6a69ba87fa63d9ee25fa4b8aeaa5f2afd73b0c63`;
- GSM8K case IDs SHA: `2b94b3596cb854898e76145401d0595e4f1a912d20cba4f7957def70051f6349`;
- local v4 report SHA:
  `8d8c48ee51c8565805095057bf676e848b5133d7dd7fdc5450454f8ef127e4c7`.

## Boundary

The preregistration reads only public reports, public case metadata, and raw
case IDs. It does not read benchmark prompts, answers, or model outputs. Raw
artifacts remain local and cannot enter training, reward, or verifier data.
After observation, no rerun or policy change is allowed.
