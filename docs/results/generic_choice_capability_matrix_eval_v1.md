# Generic Choice Capability Matrix Evaluation v1

## Result

On 32 scored fresh cases:

- anchored-v1 direct: 19/32;
- anchored-v1 plus verified executor: 25/32;
- delta: +0.1875;
- paired bootstrap 95% CI:
  [+0.0625,
  +0.3125];
- exact McNemar p=0.03125;
- fixes / regressions: 6 /
  0.

All six fixes are in explicit-average cases: direct improves from 2/8 to 8/8.
The executor makes zero overrides on 16 ambiguity cases and preserves all 16
direct outputs. Parser expected-route agreement is 48/48.

## Invalid 9B Arm

The frozen 9B direct arm is not a valid quality baseline: 48/48 outputs hit the
32-token cap while emitting reasoning, zero contain `FINAL`, and zero are
parseable. Report this as a contract/budget failure, not 9B quality accuracy.
No claim that 4B exceeds 9B is allowed from v1.

## Decision

The fresh matrix supports the verified-execution mechanism relative to 4B
direct. Pre-register a matched v2 with the same `FINAL: [A-D]` constrained
decoding for both 4B and 9B direct arms. Do not rerun or reinterpret v1's 9B
arm. The independent holdout, merge, scale, and RL remain blocked.

## Identity

- pre-registration revision: `815700d`;
- config SHA256: `f40be9cc5f1a98fb101724bae43a3ede1f477f4cf2d7a4956fa7efc42a2d7257`;
- matrix SHA256: `5db7561b95f6b951ef7fb45293e24a39276b69b5b43e04c63712f8450e37b933`;
- raw result SHA256: `fcb145d64dce23e4558d4004dea3b11f4ef1aa442e9026118abbf6966985d31b`.
