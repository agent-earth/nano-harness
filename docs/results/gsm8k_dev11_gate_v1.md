# GSM8K Dev11 Decision Gate Result

## Result

- 4B direct: 0.8333;
- 4B decision gate: 0.7500;
- 9B direct: 0.8333.

Treatment versus 4B direct is -0.0833, 95% bootstrap CI
[-0.2083,
+0.0000], with
0 treatment-only wins and
2 direct-only losses.

The gate chooses `USE_RESOLVE` on 3 cases:
0 wins,
2 losses, and
1 neutral. It emits
0 invalid raw decisions, all of
which fail closed to `KEEP`.

Treatment uses 31089 tokens and
508.8s.

## Contract Audit

Protected direct matches the independent 4B direct arm. Raw gate outputs,
decisions, selected predictions, stage inputs, and deterministic final
formatting match the committed protocol. Raw outputs remain local and ignored.

## Decision

Dev11 fails at least one directional promotion rule.

## Reproduction Identity

- Code revision: `93c768cf098d73555f864de28c1acb4deeefbabe`
- 4B direct raw SHA256: `65666db89166d5e0aba074ba3902962c75043cc2ad722b7951fdcafeba6925f0`
- 4B treatment raw SHA256: `3817f5753f03e06c91c0f576733b88a6adb957eefe2654ae7b9b89ebaea36dd6`
- 9B direct raw SHA256: `887c4bc787af5320b81fe428fca7f9324d6adacf3aa17d1af40785999e33a0e8`
