# MBPP Full-Validation Confirmation v2

This freezes the v2 policy on 47 full-validation tasks that are absent from
sanitized validation and sanitized train. It starts no model generation.

- policy: official three-shot, five candidates after direct failure, up to
  three deterministic public-test repair rounds;
- reference solution: hidden;
- sandbox: no network, read-only root, isolated Python, bounded CPU, memory,
  file size, open files, and wall time;
- case IDs SHA:
  `30687b63345feacface8f94424e19bf7af2f355212432c5f355b86d5c4e803e9`;
- config SHA: `43675163a1d4fb576404f70a13f62e74e90b7dea275f56f06fe5dc6c118db19b`;
- shard counts: `[12, 12, 12, 11]`.

Passing requires non-regression versus direct 4B and significant superiority
over matched 9B: positive paired bootstrap lower bound, exact McNemar p<0.05,
at least six candidate-only wins, and more wins than losses. Only then may the
257-case sanitized test be separately pre-registered.
