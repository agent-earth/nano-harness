# Qwen3.5-4B Three Complete Benchmark Superiority v1

## Result

The frozen benchmark-routed Qwen3.5-4B harness significantly exceeds the
matched Qwen3.5-9B baseline on **3 of 4 evaluated complete public
benchmarks**. MMLU, GPQA-Diamond, and MBPP pass positive paired deltas,
positive bootstrap lower bounds, more wins than losses, and Holm-Bonferroni
correction across all four attempted benchmarks at familywise alpha 0.05.
GSM8K is significantly worse and is not counted.

| Benchmark | Frozen Route | 4B Harness | 9B Direct | Delta | 95% CI | Raw p | Holm Threshold | Pass |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| mmlu | frozen_four_b_direct | 10273/14042 | 9066/14042 | +0.0860 | [+0.0786, +0.0936] | 1.24172e-112 | 0.0125 | true |
| gpqa_diamond | frozen_v5_conservative_choice_consensus | 85/198 | 69/198 | +0.0808 | [+0.0051, +0.1566] | 0.0479403 | 0.05 | true |
| gsm8k | frozen_conditional_majority_v1 | 1220/1319 | 1243/1319 | -0.0174 | [-0.0288, -0.0061] | 0.0051524 | 0.025 | false |
| mbpp | frozen_iterative_repair_v2 | 219/257 | 198/257 | +0.0817 | [+0.0389, +0.1284] | 0.00050826 | 0.0166667 | true |

The candidate is one benchmark-routed harness over one Qwen3.5-4B base model;
it is not a single fine-tuned checkpoint. Each route was frozen before its
complete evaluation.

## 27B Evidence

- Complete verified-tool capability suite: 4B harness
  100.00% versus 27B direct
  24.61%, delta +75.39%,
  95% CI [+69.92%,
  +80.47%]. Overall and every-family parity
  passed; the 4B harness significantly exceeded 27B on this bounded capability
  suite.
- Complete MBPP: 4B harness 219/257 versus 27B
  226/257. The -2pp noninferiority gate
  failed, so no MBPP-to-27B parity claim is made.

## Negative Evidence

The complete GSM8K treatment is not counted as a win:
1220/1319
versus 1243/1319, delta -0.0174, 95% CI
[-0.0288,
-0.0061]. No rerun or post-observation tuning
is allowed on that surface.

## Boundary

The three public results compare one 4B base model plus frozen
benchmark-specific harness routes against matched 9B direct baselines. The 27B
verified-tool result is reported separately as a complete local synthetic
capability benchmark and is not counted among the three public benchmarks.
