# Generic Choice Capability Matrix Evaluation v2

## Matched Result

All arms produce 48/48 grammar-conforming outputs. On 32 scored fresh cases:

- anchored-v1 constrained direct:
  19/32;
- 9B constrained direct: 17/32;
- anchored-v1 plus verified executor:
  25/32.

Executor versus 4B direct:

- delta +0.1875;
- 95% CI [+0.0625,
  +0.3438];
- McNemar p=0.03125;
- six wins and zero losses.

Executor versus 9B:

- delta +0.2500;
- 95% CI [+0.0938,
  +0.4062];
- McNemar p=0.00781;
- eight wins and zero losses.

4B direct versus 9B is not significant: delta
+0.0625, CI
[-0.0625,
+0.1875], p=
0.625.

## Safety And Scope

The executor makes zero overrides on all 16 ambiguity cases and preserves
their constrained 4B outputs. Expected-route agreement is 48/48.

This is significant superiority on a fresh generic capability matrix, not on
the three benchmark suite and not independent holdout evidence. The matrix is
ineligible for every training use. Independent holdout, merge, scale, and RL
remain blocked.

## Identity

- pre-registration revision: `52945f0`;
- config SHA256: `47776628053eb2f3fa34e2a6cfc7b6682734a702a03aab8bff834c48d11d9c6c`;
- matrix SHA256: `5db7561b95f6b951ef7fb45293e24a39276b69b5b43e04c63712f8450e37b933`;
- 4B model config SHA256:
  `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`;
- 9B model config SHA256:
  `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`;
- raw result SHA256: `b9cc50b51bc6b9bd5c8b0d64bd72a138a7e3e97482a5a3ba891d589719e7f322`.
