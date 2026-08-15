# GSM8K Dev12 Deterministic Majority Result

## Result

- 4B direct: 0.9583;
- 4B deterministic majority: 0.9583;
- 9B direct: 0.9583.

Treatment versus 4B direct is +0.0000, 95% bootstrap CI
[+0.0000,
+0.0000], with
0 treatment-only wins and
0 direct-only losses.

A numeric majority exists on 16 cases; no majority
keeps direct on 8. Voting changes the direct answer
on 0 cases:
0 wins, 0 losses, and
0 neutral.

Treatment uses 33493 tokens and
865.8s.

## Contract Audit

Protected direct matches the control, both isolated re-solve inputs match the
committed prompts, and every majority/count/selection is recomputed from
recorded normalized predictions. Raw outputs remain local and ignored.

## Decision

Dev12 fails at least one directional promotion rule.

## Reproduction Identity

- Code revision: `6a75f967f079989dc8d03d78a3bd95021b4d78c2`
- 4B direct raw SHA256: `93969cc10a1a85b602b4be0bccd473807d2a7acbef56d5792ae99a7dd7ada6fa`
- 4B treatment raw SHA256: `7fa5c149029c38634dc4e0c41b1023e8d2a1397220af395dd7816af1da0c034c`
- 9B direct raw SHA256: `da478693a4460f2b8b49278f5a5b441ca2d8fc38b3eb4f38524abb293fdf657d`
