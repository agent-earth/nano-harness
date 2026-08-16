# GSM8K Dev16 Constrained Recovery Confirmation

## Result

- 4B direct: 0.9479
  (91/96);
- 4B constrained recovery: 0.9479
  (91/96);
- 9B direct: 0.9479
  (91/96).

Treatment versus 4B direct is +0.0000, with paired bootstrap
95% CI [+0.0000,
+0.0000] and exact McNemar
`p=1`.

Treatment versus 9B direct is +0.0000, with paired bootstrap
95% CI [-0.0417,
+0.0417] and exact McNemar
`p=1`. There are
2 treatment-only wins and
2 9B-only wins.

Recovery triggers 0 times, produces
0 correct recoveries, and leaves
0 unparseable. All
0 recovery outputs full-match the
committed regex. Treatment/direct token ratio is 1.000x.

Recovery remains unobserved despite doubling the fresh slice, so dev16 cannot establish mechanism benefit.

## Case-Level Evidence

- Recovery cases: none
- Treatment-only wins versus 4B direct: none
- Direct-only wins versus treatment: none
- 4B direct parse failures:
  []
- 9B direct parse failures:
  ["gsm8k-41241715b3dd1218", "gsm8k-661c711a549d463d", "gsm8k-6661641947ff80d4", "gsm8k-92c99a410d1a454e"]

## Contract Audit

All arms contain the committed 96 case IDs. Direct stage input hashes match
the committed prompts. Treatment invokes recovery exactly for direct parse
failures, preserves every parseable direct prediction, carries the committed
`structured_outputs.regex`, full-matches each recovery output, and selects the
recorded result deterministically. Raw outputs remain local and ignored.

## Cost

- 4B direct: 32430 tokens,
  809.5s summed request latency,
  0 API errors;
- 4B treatment: 32430 tokens,
  806.2s summed request latency,
  0 API errors;
- 9B direct: 31735 tokens,
  763.6s summed request latency,
  0 API errors.

## Decision

Dev16 fails at least one pre-registered directional promotion rule.

Stop enlarging GSM8K development windows; retain the validated mechanism as an optional no-op parse guard and return to higher-leverage harness hypotheses.

## Reproduction Identity

- Pre-registration/code revision: `3b7d835bd5acac1019867d7e37ace15cc39a7f32`
- 4B direct raw SHA256: `5300a96906bd43893af1b4aa5144a3c4e1a198ad356085b4589e5e7f894f38e4`
- 4B treatment raw SHA256: `49eccdab57a2ada4bd26df87833df1f442b6e437b924778eb588b01742cbaa92`
- 9B direct raw SHA256: `3907143f3567d68d9b83dfd0f71684e0ac2260d3e9d430ffa3e4ebccfa7aee0f`
