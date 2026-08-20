# Qwen3.5 Constrained Semantic Model Router v1

## 目标

Exact markers 在真实 question surface 上 0/15559 覆盖。下一机制不从这些已扫描
prompts 提炼规则，而是在 fresh non-benchmark surface 上验证 4B enum router：

- `implicit_scale_total`
- `first_strict_profit_period`
- `NONE`

## Fresh Surface

- positive 128：两个 semantic families 各64，使用自然改写且不含旧 exact markers；
- negative 128 unsupported：box total / remaining stock 各64，必须全部选
  `NONE`；
- prior/benchmark prompt overlap：0；
- benchmark/canary/holdout rows 或 outputs：0。

## Frozen Pipeline

1. 4B router 只输出三个枚举之一，16 tokens；
2. `NONE` 直接复用 direct；
3. 正类只暴露一个 parent typed semantic skill；
4. planner、source validator、executor、feedback equality 与 fallback 保持不变；
5. router 或 planner 都不读 expected、correctness 或 case metadata。

## Gate

- 256 router outputs parseable；
- positive route recall 128/128；
- negative NONE 128/128，false positive 0；
- 128 positive verified executions 和 feedback matches；
- negative candidate 与 direct 评分字段完全一致；
- candidate vs 4B / 9B 均显著、至少12 wins、0 losses；
- every-family non-regression。

通过只允许另行预注册 **question-only real router scan**，不能直接生成 benchmark
outputs，也不能重跑 canary、访问 holdout 或训练。

## Boundary

- config SHA：`8c6f0215cb2a0f805e04fe6c00a28fc3b5847d0c2a14575a90b3ed2a6586ebce`；
- case contract SHA：`6bb1f7ff3079fa1e3c75d033e26d3c6f6109b0abdf96c8949edebea2761c31ca`；
- model generation：false；
- evaluation started：false；
- benchmark/canary/holdout accessed：false。
