# V11 Adapter Regression Canary Result

## Result

The unchanged v11 adapter passes the sealed regression canary:

- GSM8K: 15/16, threshold 14/16;
- MMLU: 13/16, threshold 13/16;
- GPQA-Diamond: 4/8, threshold 3/8;
- total: 32/40, threshold 30/40;
- API errors: 0;
- parse failures: 0;
- length truncations: 0.

The calibration remains base 4B 30/40 and rejected v6
28/40. Namespace conversion preserves all 224 tensor contents,
the adapter parent is correct, and base/adapter logits differ.

## Boundary

This post-v6-calibrated canary is a regression gate only. It cannot establish
quality uplift and its case-level outputs or IDs must not enter training.

Passing permits only the exact adapter to run the frozen 211-case matched
suite. Merge, scale-up, and RL remain forbidden.

## Identity

- manifest SHA256: `e213985897e1da260c24e8f383e80d02a3f9c880a09f45e6cc2cc27f51dcf0f8`;
- candidate raw SHA256: `85d993405f6f7813e4dfaba2b813719fb3b0abda6558f3827aec463ed591a4cb`;
- namespace receipt SHA256:
  `c19ad8955b93dc01a924f9d8eebccf2ae25322c830c67096824db09b6d648fe3`;
- serving parity SHA256: `ebc89761a0866007894d351d6ec663c6a9ffdd9f0ecfa6ddc78cda69f8d6aec9`;
- local gate report SHA256:
  `537a981012f2a9f52653bb881f743a966bd8ed9b8b92cc17dc77922f5a8265b6`.
