# V11 Full Matched Adapter Result

## Task Results

| Benchmark | V11 adapter | Base 4B | 9B |
| --- | ---: | ---: | ---: |
| gsm8k | 89/96 | 90/96 | 89/96 |
| mmlu | 66/96 | 67/96 | 58/96 |
| gpqa_diamond | 7/19 | 6/19 | 4/19 |

V11 scores 162/211, versus base 4B
163/211 and 9B 151/211.

## Candidate Versus Base 4B

- candidate macro: 0.6610;
- base 4B macro: 0.6504;
- micro delta: -0.0047;
- paired 95% CI:
  [-0.0284,
  +0.0190];
- exact McNemar p: 1.000000;
- task non-regression: False;
- parse non-regression: True.

## Candidate Versus 9B

- candidate macro: 0.6610;
- 9B macro: 0.5806;
- micro delta: +0.0521;
- paired 95% CI:
  [-0.0095,
  +0.1137];
- exact McNemar p: 0.135156;
- task non-regression: True.

The 11-case point improvement over 9B is not statistically supported because
the confidence interval crosses zero and McNemar p is above 0.05.

## Failure Diagnostic

Candidate GSM8K has 7/96 official failures,
including 2 parse failures and
2 length truncations. The official score is
unchanged by diagnostics.

Against base 4B, there are
3 candidate-only wins and
4 base-only wins. The latter
are the bounded source for abstract failure-family analysis; their benchmark
rows remain ineligible for training.

## Decision

V11 fails base-4B task non-regression because GSM8K and MMLU are each one case
lower. It also fails the pre-registered statistical superiority gate versus
9B. Reject promotion, merge, scale-up, and RL.

Preserve the GPQA gain, local family improvement, and format stability. The
next data ablation may use only abstract failure families, never benchmark or
canary prompts, outputs, references, or case payloads.

## Reproduction Identity

- pre-registration revision: `6ccc260`;
- candidate raw SHA256: `6b684636311e896aa7b6394ab77f0547d6c21715a39e21b414e6ac252075f444`;
- base 4B raw SHA256: `c59383d3fd3d6087025d6e1ff649979d9d5a9e8dc73b5429a4f8e9fa41b6b8c7`;
- 9B raw SHA256: `ffae93774d51b87a2e29258d170a84f8b165f996e2e78eedd102271dfc260044`;
- adapter tree SHA256: `87248908918b06c2d28ff68efd4f0b1ff92ca8bf8b7588e1c7e81a85eb7da852`;
- namespace receipt SHA256:
  `c19ad8955b93dc01a924f9d8eebccf2ae25322c830c67096824db09b6d648fe3`;
- serving parity SHA256: `ebc89761a0866007894d351d6ec663c6a9ffdd9f0ecfa6ddc78cda69f8d6aec9`;
- canary report SHA256: `5622f50f7207657d5923b9309943cb8572c3e16847210ec1368c15baf5ff4345`.
