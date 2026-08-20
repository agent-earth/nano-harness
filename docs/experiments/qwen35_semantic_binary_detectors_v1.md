# Qwen3.5 Semantic Binary Detectors v1

## 设计

Multiclass router 对 unsupported tasks 0 false positive，但 implicit-scale recall
为0。新实验拆成两个独立 detector：

- implicit-scale detector：`DETECT: YES/NO`
- strict-profit detector：`DETECT: YES/NO`

组合规则固定：恰好一个 YES 才选该 skill；双 NO 或双 YES 一律 `NONE` 并
direct-preserve。

## Fresh Surface

- 128 cases：4 families × 32；
- positive 64，negative 64；
- prior/benchmark prompt overlap：0；
- benchmark/canary/holdout rows 或 outputs：0。

## Gate

- 256 detector calls 全部 parseable；
- composition correct 128/128；
- positive recall 64/64；
- negative NONE 64/64，false positive 0；
- conflict 0；
- positive 64 verified executions 与 feedback matches；
- negative candidate 与 direct 完全一致；
- candidate vs 4B/9B 均显著、至少12 wins、0 losses。

通过也只允许另行预注册 question-only real detector scan，不允许 benchmark
generation、canary rerun、holdout 或 training。

## Boundary

- config SHA：`32c1d877a1cecdb9041fc88226fa2e52890390712f449a8552be0a1202d88748`；
- case contract SHA：`a3993974c923b5cab4b8570642226076a8b9c51ead24175773681a8c84b624a5`；
- model generation：false；
- evaluation started：false；
- benchmark/canary/holdout accessed：false。
