# GSM8K Dev15 Constrained Recovery Result

## Result

- 4B direct: 0.9583;
- 4B constrained recovery: 0.9583;
- 9B direct: 0.9792.

Treatment versus 4B direct is +0.0000, 95% bootstrap CI
[+0.0000,
+0.0000], with
0 treatment-only wins and
0 direct-only losses.

Recovery triggers 0 times, produces
0 correct recoveries, and leaves
0 unparseable. All
0 recovery outputs match the
committed regex.

Treatment token ratio versus direct is 1.000x.

No direct parse failure occurs in this 48-case slice, so recovery never fires.
The structured-output capability is validated by a real smoke request and
audited implementation, but this slice cannot establish recovery benefit.

## Contract Audit

Recovery calls exist exactly for direct parse failures. Parseable direct
predictions are unchanged. Every recovery includes the committed structured
output metadata and full-matches the numeric FINAL regex. Raw outputs remain
local and ignored.

## Decision

Dev15 fails at least one directional promotion rule.

The next experiment expands to 96 fresh GSM8K cases without changing policy.

## Reproduction Identity

- Code revision: `a08f340f8f94d5f16e68c0886296fd4fe17522e9`
- 4B direct raw SHA256: `dee4303f90e90797ac1352599392f109f1988de7016803fb2870e7f7ab13c006`
- 4B treatment raw SHA256: `3ce56f301b790c1ae8efb449df5115aad9ccccd3b07b32cec90456351182bc56`
- 9B direct raw SHA256: `a16b2ee1666f9404402dedb572d3cfaed80cb432c094664bbd232a238685125e`
