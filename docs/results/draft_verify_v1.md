# Draft-Verify v1 Result

## Decision

The treatment is retained as a promising harness component, but it does not
satisfy the harness-stage acceptance criterion. On the fixed 72-case suite,
4B draft-verify scores 0.7222 versus the
9B baseline at 0.7083, a
+0.0139 macro delta. The paired micro 95% bootstrap
interval is [-0.0833, +0.1111], so the lead is not significant.

## Disjoint Development Slice

- Direct 4B macro: 0.2778
- Draft-verify 4B macro: 0.5000
- Paired delta: +0.2222
- Paired counts:
  `{'both_correct': 5, 'candidate_only': 4, 'baseline_only': 0, 'both_wrong': 9}`
- Parse failures: 13 direct to
  0 treatment
- Tokens: 5563 direct to
  10331 treatment

The 18 development cases are disjoint from the fixed 72 evaluation cases.

## Fixed Evaluation

| Benchmark | 4B direct | 4B draft-verify | 9B direct |
| --- | ---: | ---: | ---: |
| GSM8K | 0.9167 | 1.0000 | 0.9167 |
| MMLU | 0.7500 | 0.8333 | 0.7917 |
| GPQA-Diamond | 0.4167 | 0.3333 | 0.4167 |
| Macro | 0.6944 | 0.7222 | 0.7083 |

Draft-verify improves GSM8K and MMLU by two cases each versus 4B direct, but
loses two GPQA cases. It uses about twice the tokens while reducing wall-clock
time in this single-sequence local setup. The next iteration should preserve
the verifier for math/knowledge tasks and test a GPQA-specific repair on a new
development slice, not tune again on the observed fixed evaluation cases.

## Reproduction Identity

- Code revision: `cf2a00e1d703747a754e85596f80808c76b5fb3c`
- Dev control manifest SHA256: `b47a935907cbd2a87236691742143c9c79741273c972d4091901fb0d6d0cb59f`
- Dev treatment manifest SHA256: `02333ef588f848189ec77c8260189f6ea08fec648719005cf1a65c392e109445`
- Eval treatment manifest SHA256: `e19b782fb5316e207380d758694db5e72e9ee08453b3b687ee7b6f99ea9adda8`
- Dev direct raw SHA256: `abc64deda159c1af9ee3bc748919f631a9874b4c95435dd3ee69c432563bfc7f`
- Dev treatment raw SHA256: `0c351f29fb8f02a288cbbaec3eabb4b667b98f4f3b990b723b2b8c5c10b35327`
- Eval treatment raw SHA256: `a2e0ba00bd6ca3e26288a82e227355a420039c8416c8ad76fbfc5b49a4ee4d9d`

Raw outputs remain local and ignored. The public JSON contains metrics,
artifact digests, and case IDs only.
