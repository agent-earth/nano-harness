# Anchored-v1 Verified Choice Canary v1 Result

## Result

The exact verified-choice executor passes the old sealed canary:

- GSM8K: 15/16;
- MMLU: 13/16;
- GPQA-Diamond:
  4/8;
- total: 32/40;
- API errors, parse failures, and truncations: 0 / 0 / 0.

All 24 choice prompts are outside parser v1's explicit arithmetic-average
intent. The executor therefore performs zero overrides and reuses 40/40
anchored-v1 outputs byte-for-byte. There are zero regressions.

## Boundary

This post-v6-calibrated canary remains a regression gate only. It is not
independent quality evidence and no case-level payload is published or
training eligible.

Passing permits only the old 211-case development suite. The independent
holdout, merge, scale, and RL remain blocked.

## Identity

- pre-registration revision: `8ffba19`;
- config SHA256: `3cc141f3b1505f7e9fd222c7fad21e4f220ece8ed16e73952d3c67e082a36c2b`;
- canary manifest SHA256: `e213985897e1da260c24e8f383e80d02a3f9c880a09f45e6cc2cc27f51dcf0f8`;
- anchored-v1 raw SHA256: `c42b266b234b3f77f9b22ff8f70e5f2c7de6503a9fa96282b7fcfa4522c74b91`;
- local pass receipt SHA256: `f226f12956d3a72fdc3f6923a069c1786dbfa1e08496e4502eb2561a424009c2`;
- raw applicator result SHA256: `a80be97d4dd609588b0d29771a6be738260e8265d9d621227363f7bb4ac08e46`.
