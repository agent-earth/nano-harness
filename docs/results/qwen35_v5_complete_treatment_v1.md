# Qwen3.5 V5 Complete Treatment v1 Result

## Verdict

**REJECT.**

| Benchmark | Candidate | Direct 4B | Direct 9B | Delta vs 9B | 95% CI | McNemar p |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| gsm8k | 1156/1319 | 1203/1319 | 1240/1319 | -0.0637 | [-0.0811, -0.0478] | 6.38467e-14 |
| mmlu | 10273/14042 | 10273/14042 | 9066/14042 | +0.0860 | [+0.0788, +0.0933] | 1.24172e-112 |
| gpqa_diamond | 85/198 | 76/198 | 69/198 | +0.0808 | [+0.0051, +0.1566] | 0.0479403 |

## What The Treatment Did

- GSM8K ran three target-blind grounded-expression attempts for every case.
  It overrode direct 4B on 474 cases, but
  changed the parsed answer on only 84 cases.
  Those changes produced
  14 wins and
  61 losses versus direct
  4B, for a net
  -47 correct answers.
- MMLU preserved the frozen direct 4B output for all 14,042 cases and made no
  model calls, so its gain over 9B is the already-established direct-model
  advantage rather than a treatment gain.
- GPQA used two independent reviews and required a confirming third call before
  replacing a non-direct choice. It overrode and changed
  37 cases, producing
  16 wins and
  7 losses versus direct
  4B, for a net +9.

## Conclusion

The conservative GPQA choice consensus transferred: it improved direct 4B by
9 correct answers and significantly beat 9B. The GSM8K calculator consensus did
not transfer: repeated agreement mostly amplified the same wrong expression,
causing a statistically significant regression versus both direct 4B and 9B.
The complete treatment is therefore rejected even though two of three
benchmarks beat 9B.

## Gates

```json
{
  "all_rows_complete": true,
  "all_three_complete_benchmarks_won": false,
  "gpqa_diamond_non_regression_vs_four_b": true,
  "gpqa_diamond_superior_to_nine_b": true,
  "gsm8k_non_regression_vs_four_b": false,
  "gsm8k_superior_to_nine_b": false,
  "mmlu_non_regression_vs_four_b": true,
  "mmlu_superior_to_nine_b": true
}
```

No rerun, prompt, route, parser, budget, consensus, or scorer change is allowed
after this complete result.
