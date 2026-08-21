# MBPP Sanitized Iterative Repair Train v2

This freezes one method-development run on all 120 sanitized train tasks. It
starts no generation and does not reopen the observed validation split.

## Changes From v1

- use the official three-shot examples with task IDs 2, 3, and 4;
- accept the first Python fence anywhere, `[BEGIN]`/`[DONE]`, or parseable
  plain Python;
- generate five independent candidates after a direct failure;
- perform up to three deterministic repair rounds from the current best
  candidate;
- show repair the failed public-test indices and failure classes;
- preserve passing direct 4B and override only on a strictly higher public-test
  pass count.

The evaluation reference implementation remains hidden. Generated code runs
in the same no-network, read-only-root bubblewrap sandbox.

## Identity

- config SHA: `4b7173012a9ffe07a3e5964d609ada9019a7d3c7099242045109c943244b67d8`;
- train SHA: `d95f8ad6d2fff08fe4826122d6e3e31f75716825d0c5c340d297aca5e9e0de0e`;
- train case IDs SHA: `cbfb6f66d4c41ce3ffa242b39b5f37312bdfa93959583a5ae914849d67218514`;
- frozen v1 report SHA: `498775115585bd5f7f8edbd43ef842b9d86a9e5b91d9acb5c4380e7f35fd65a5`.

The 43-case validation v1 and 257-case test are not generated or scored here.
No post-observation tuning or rerun is allowed on this train surface.
