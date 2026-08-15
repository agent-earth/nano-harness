# GSM8K Confirmation Result

## Result

On 96 unseen GSM8K cases, 4B draft-verify scores
0.8750 and 9B direct scores
0.9375. The paired delta is
-0.0625, bootstrap 95% CI [-0.1146, -0.0208],
exact McNemar `p=0.031250`.

The pre-registered confirmation fails:

- the 4B point estimate is lower than 9B;
- the CI lower bound is below the -0.05 non-inferiority margin;
- six cases are 9B-only wins and none are 4B-only.

## Failure Mechanism

Among the six 9B-only cases:

- 4 4B drafts hit the 256-token limit;
- 2 4B drafts stopped with incorrect reasoning;
- the strict verifier corrected 0 to the reference.

The verifier reliably formats but does not independently repair math reasoning.
The next experiment must use fresh dev3 cases and an independent math re-solve
verifier. No tuning is allowed on these 96 confirmation cases.

## Cost And Identity

- 4B draft-verify: 58561 tokens,
  655.0s.
- 9B direct: 31134 tokens,
  723.4s.
- Code revision: `d6ee71ccdccd9b3b43d39572eeae1438f90e9284`
- 4B raw SHA256: `90c6b84a0ddc1f388e15fbac45cff83476586cebcd925a650ab031b9af88257c`
- 9B raw SHA256: `4f26a86c724cde2ca60b4036101975f6df93b569634b7891b26fdd2882dee7f4`

Raw prompts and outputs remain local and ignored.
