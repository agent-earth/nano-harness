# Generic Choice Verifier Matrix v2 Result

## Matched Result

All arms are 48/48 parseable. On 16 scored fresh exact cases:

- anchored-v1 constrained direct:
  8/16;
- 9B constrained direct: 11/16;
- anchored-v1 plus verifier v2:
  16/16.

Executor versus 4B direct:

- delta +0.5000;
- 95% CI [+0.2500,
  +0.7500];
- McNemar p=0.00781;
- eight wins and zero losses.

Executor versus 9B:

- delta +0.3125;
- 95% CI [+0.1250,
  +0.5625];
- McNemar p=0.06250;
- five wins and zero losses.

The CI is positive, but p=0.0625 misses the pre-registered 0.05 dual
significance criterion. Do not claim significant superiority over 9B.

## Safety And Scope

Verifier v2 scores 8/8 on host-count and 8/8 on verbal-average exact cases.
It makes zero overrides on all 32 ambiguity cases and preserves all 32 direct
outputs. Expected-route agreement is 48/48.

This is fresh generic mechanism evidence, not benchmark or holdout evidence.
The matrix is forbidden for every training use. Holdout, merge, scale, and RL
remain blocked.

## Decision

Pre-register a larger history-disjoint exact-case replication to test the 5-0
trend with enough McNemar power. Preserve ambiguity families as safety-only;
do not train on any matrix row.

## Identity

- pre-registration revision: `5a38882`;
- config SHA256: `00e5b2125edee0c8e54c771746952375d6227bb69aa518bb195b642e61dd1ef1`;
- matrix SHA256: `70330f730f144c1fb05d50a27e566321561451a962d0bdf736f27b8faa2f79b0`;
- raw result SHA256: `e6b208e1b3d816fc540d6a7c7e9315338751c09804b248691c633186491c4c87`.
