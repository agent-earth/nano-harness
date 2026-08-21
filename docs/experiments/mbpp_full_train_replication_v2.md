# MBPP Full-Train Replication v2

This freezes the unchanged v2 harness on 254 full-train tasks absent from
sanitized train. It starts no model generation and is not a test score.

- policy: byte-for-byte-equivalent model, prompt, parser, direct, candidate,
  and sandbox sections from the frozen v2 config;
- reference solution: hidden;
- prior overlap: zero with sanitized/full validation, sanitized test, and
  official few-shot rows;
- case IDs SHA:
  `f3ad3ba8bfd7fd47e08e5db94c8a2de7e1ba5299e8d0396bb526725a79fae93a`;
- config SHA: `5bfd75fb03250cd2385090103e41902f1ab26aeda631eadb903463f2ed961139`;
- shard counts: `[32, 32, 32, 32, 32, 32, 31, 31]`.

Passing requires non-regression versus direct 4B and significant superiority
over matched 9B: positive paired-bootstrap lower bound, exact McNemar p<0.05,
at least six candidate-only wins, and more wins than losses. Only then may the
257-case sanitized test be separately pre-registered.
