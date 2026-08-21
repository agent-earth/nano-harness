# Verified-Tool 27B Parity v1

This freezes one complete 256-case, four-family parity comparison before any
27B evaluation generation.

- frozen 4B harness: 256/256, raw SHA
  `3d4c987f8f949289e50d97bdb7f00dd08036eec511a8d77261cc0b24ddbb8047`;
- 27B arm: direct constrained generation from the validated BF16 TP=2 service;
- case contract SHA: `cd5037a2574254ad87ff13abc3d8af51670d9b565d4433b041df4968c5eb2d71`;
- config SHA: `71d0b98a6f7d857fba48588ce81ec0aed0d97f5862930869bcc283193c7f57f2`;
- shard counts: `[128, 128]`;
- noninferiority margin: 2 percentage points.

Parity requires the paired-bootstrap 95% lower bound to be at least -0.02 both
overall and in every one of the four families. The run is one-shot, and it
uses no benchmark rows or outputs.
