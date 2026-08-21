# MBPP Sanitized Test v2

This freezes one complete 257-case MBPP sanitized-test run. It starts no model
generation.

- policy: unchanged from the admitted v2 development and replication runs;
- reference solution: hidden;
- overlap with sanitized train, validation, and few-shot rows: zero;
- test case IDs SHA: `4b7e1f74447041a3f9ad4c02a27c157fd1ef98beb2c770796f4bfd8e630863d8`;
- config SHA: `f37ab18661ce84a8a7adec664ae8a6114ca745266108099a77916b37528bdc5b`;
- shard counts: `[33, 32, 32, 32, 32, 32, 32, 32]`.

A complete-benchmark superiority claim requires non-regression versus direct
4B and significant superiority over matched 9B under every pre-registered
gate. The test is run once; no rerun or post-observation tuning is allowed.
