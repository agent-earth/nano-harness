# MBPP Sanitized Verified Selection Dev v1

This pre-registers a full 43-case validation run and starts no model
generation. One test row was previously inspected only for schema feasibility;
it did not determine this policy. The 257-case sanitized test generation
remains closed.

## Candidate

- Run matched direct Qwen3.5-4B and Qwen3.5-9B once per task.
- If direct 4B passes every public test, preserve it without extra calls.
- Otherwise generate three independent 4B candidates.
- Include the public MBPP `test_list` in the model prompt, matching the
  benchmark protocol; keep reference solutions hidden.
- Execute candidates against those tests in a no-network, read-only-root
  bubblewrap sandbox.
- Select by public-test pass count, then shorter code, then replica index.
- If none passes all tests, allow one repair using only aggregate pass count
  and failure classes; reference code remains hidden.
- Override direct 4B only when the selected candidate passes more public tests.
- Run four deterministic shards assigned by sorted case index modulo four;
  merge requires the exact 43-case set.

## Identity

- config SHA: `8e02b0adc5ca78a4a197ec6622bcc65a0b30de3b6f9cdb64aa6446ce453775c2`;
- validation data SHA: `27e065fcab3c863959933328a7fdbf404e1bcb5464b1be6fe0dcd9530e420204`;
- validation case IDs SHA:
  `f73a3315369d6de7bff3596e234f56a1cc2c3d8b82cc6c495eedf8fc4b828ebe`;
- cases: 43.

## Gate

Validation must preserve 4B with a non-negative paired bootstrap lower bound
and show a positive directional delta over matched 9B with more wins than
losses. Only then may the 257-case sanitized test be separately
pre-registered. No post-observation tuning or validation rerun is allowed.
