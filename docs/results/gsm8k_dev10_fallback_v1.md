# GSM8K Dev10 Protected Fallback Result

## Result

- 4B direct: 0.8333;
- 4B protected fallback: 0.8333;
- 9B direct: 0.8750.

Treatment versus 4B direct is +0.0000, 95% bootstrap CI
[+0.0000,
+0.0000], with
0 treatment-only wins and
0 direct-only losses.

Fallback fires 10 times. The independent re-solve
disagrees with direct on 6 cases; the
arbiter makes 0 parseable overrides:
0 wins, 0
losses, and 0 neutral.

Within the fallback cases, protected direct is correct on
7 and the re-solve is correct on
7. There are
2 cases where
protected is wrong but re-solve is correct, so unconditional protected
fallback discards usable repair evidence.

Treatment uses 29810 tokens and
483.9s.

## Contract Audit

Protected direct matches the independent 4B direct arm. Every fallback occurs
exactly when raw arbiter output is unparseable and protected direct is
parseable; final fallback predictions equal protected direct. Raw outputs
remain local and ignored.

## Decision

Dev10 fails at least one directional promotion rule.

The next fresh experiment separates decision from formatting: a short gate
must output only `KEEP` or `USE_RESOLVE`, then deterministic code emits the
selected numeric `FINAL:` line. No dev10 output is rescored.

## Reproduction Identity

- Code revision: `5eefa0504a5fe3a646f612d2b34d7df1303a1222`
- 4B direct raw SHA256: `a263ce296589364db31be3360e6dfd193b6a3bb0f4ff2b6d12127d77cf466cb5`
- 4B treatment raw SHA256: `7d11df79cf7db1d095db2aae3764a3d8a9768952037bb0ae49b21136db6c7dbb`
- 9B direct raw SHA256: `460d27a5de0242964a39b8832319b75555abdb85e4828c674ad5f8f24d2fe960`
