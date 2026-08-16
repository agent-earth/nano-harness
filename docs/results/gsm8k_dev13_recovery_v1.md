# GSM8K Dev13 Conditional Recovery Result

## Result

- 4B direct: 0.8125;
- 4B conditional recovery: 0.8125;
- 9B direct: 0.9167.

Treatment versus 4B direct is +0.0000, 95% bootstrap CI
[+0.0000,
+0.0000], with
0 treatment-only wins and
0 direct-only losses.

Recovery triggers 4 times, produces
0 correct recoveries, and leaves
4 unparseable. Parseable direct outputs
are unchanged.

Treatment uses 20049 tokens versus
17784 for direct, ratio 1.127x.

All 4 recovery calls reach the 384-token limit and
remain unparseable. No recovery win is observed.

## Contract Audit

Recovery calls exist exactly for direct parse failures. Every parseable direct
prediction is unchanged. Recovery input hashes and deterministic selections
match the committed protocol. Raw outputs remain local and ignored.

## Decision

Dev13 fails at least one directional promotion rule.

Fresh dev14 keeps conditional triggering and protected-direct invariance, but
uses a 64-token answer-only recovery that solves internally and emits only the
numeric final. No dev13 output is rescored.

## Reproduction Identity

- Code revision: `9e496927a22004664320efe7741cdfaeaf63d9e8`
- 4B direct raw SHA256: `47a6f1b415c062b00ec4c4b68cebc7fc63c01d78e3969dfae4a574937c43f4f9`
- 4B treatment raw SHA256: `c16d551e1bf1d6b471e1d91809ec95274cb2f60ee10d6dcf9e022bbfb28b9ab3`
- 9B direct raw SHA256: `2019580f5a524f366acb49c59389660dda128621ff65084d55244f16c862ed79`
