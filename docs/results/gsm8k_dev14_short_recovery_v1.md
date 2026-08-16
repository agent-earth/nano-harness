# GSM8K Dev14 Short Recovery Result

## Result

- 4B direct: 0.9375;
- 4B short recovery: 0.9375;
- 9B direct: 0.9375.

Treatment versus 4B direct is +0.0000, 95% bootstrap CI
[+0.0000,
+0.0000], with
0 treatment-only wins and
0 direct-only losses.

Recovery triggers 1 times, produces
0 correct recoveries, and leaves
1 unparseable. Parseable direct outputs
are unchanged.

Treatment uses 16860 tokens versus
16623 for direct, ratio 1.014x.

The only recovery derives the correct intermediate count but spends all 64
tokens explaining it and truncates before `FINAL:`. Prompt-level answer-only
instruction is insufficient.

## Contract Audit

Recovery calls exist exactly for direct parse failures. Every parseable direct
prediction is unchanged. Short-recovery inputs and deterministic selections
match the committed protocol. Raw outputs remain local and ignored.

## Decision

Dev14 fails at least one directional promotion rule.

Fresh dev15 keeps conditional triggering and uses service-enforced constrained
decoding so recovery can emit only a numeric `FINAL:` line. No dev14 output is
rescored.

## Reproduction Identity

- Code revision: `5eaebf759ad2bec4af2c47593a7d28794c16e9d6`
- 4B direct raw SHA256: `4814afaeb04227599ab6cbc7765d2ca5f2caf2070138ce511c1eef7ca4f0289f`
- 4B treatment raw SHA256: `d0f862b986d0dba57af32b77d474b56dcb09c5d327fb11ddb7adbb9c0b6dea2e`
- 9B direct raw SHA256: `40d4d3e5a251bc4b7cd09a64763889e5070d862783c22e8a1ec1ca85b6698d31`
