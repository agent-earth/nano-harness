# Generic Choice Exact Replication v3 Result

## Matched Result

All arms are 32/32 parseable on 32 fresh scored exact cases:

- anchored-v1 constrained direct:
  22/32;
- 9B constrained direct: 11/32;
- anchored-v1 plus verifier v2:
  32/32.

Verifier v2 versus anchored-v1 direct:

- delta +0.3125;
- 95% CI [+0.1562,
  +0.4688];
- McNemar p=0.00195312;
- 10 wins and 0 losses.

Verifier v2 versus 9B:

- delta +0.6562;
- 95% CI [+0.5000,
  +0.8125];
- McNemar p=0.0000009537;
- 21 wins and 0 losses.

This passes the pre-registered dual significance criterion and the minimum
six-win, zero-loss discordance criterion.

## Family Result

Verifier v2 is 16/16 on host-count and 16/16 on verbal-average. Anchored-v1
direct is 8/16 and 14/16; 9B direct is 4/16 and 7/16, respectively.

All 32 verifier receipts reproduce byte-for-byte from the unchanged
target-blind parser. Every route is a unique exact proof; references are not
passed to the parser.

## Scope

This replicates significant verifier-v2 superiority over matched 9B for these
two exact generic mechanisms. It is not GSM8K, MMLU, GPQA, agent-benchmark, or
independent-holdout superiority. The matrix is forbidden for every training
use. Holdout, merge, scale, and RL remain blocked.

## Decision

Preserve verifier v2 as replicated mechanism evidence. The next intervention
must be benchmark-agnostic yet capable of transferring to the frozen local and
prior three-task suites, with per-task non-regression before holdout access.

## Identity

- pre-registration revision: `3b90f87`;
- config SHA256: `c71f0cd6f730946fed6c92c5d2172bbbf8a03c937127de1e9d975d8134270ace`;
- matrix SHA256: `0962a82e02151a7af1f3b498bab0f50d8004e630e851401687084e4d77fc8276`;
- raw result SHA256: `2720c866ab72090b15feb3ffedd2fc6f66890cac118ad8ab8afb21230fb65c49`.
