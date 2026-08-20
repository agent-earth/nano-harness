# Qwen3.5 Typed Semantic Skill Execution v1

## 假设

上一轮 literal calculator canary 保住了 209 行，但两个 recovery 都没变成新增
正确项。失败不是安全执行器算错，而是 tool semantics 太弱：

- `double` 是语言算子，不是题面数字；
- break-even 12 与“第一个严格盈利的整数周期”13不是同一语义。

本实验只在 fresh synthetic local surface 验证两个 typed semantic skills：

1. `implicit_scale_total`：executor 内部把 `double/triple` 映射为 2/3；
2. `first_strict_profit_period`：executor 计算
   `floor(setup_cost / period_net) + 1`，显式实现严格正收益边界。

Router 只读 prompt marker，不读 case family metadata、expected 或 correctness。
每行只暴露一个 skill schema。

## Fresh Surface

- 256 cases，2 families × 128；
- case contract SHA：
  `913c64a4bcfdce4d04299b9958a291937d3c7bfd6425ee956e835042e29651bf`；
- prior choice/tool surface overlap：0；
- complete GSM8K/MMLU/GPQA prompt overlap：0；
- benchmark outputs、canary rows/outputs、holdout rows：0；
- training eligible rows：0。

## Frozen Gate

- 256/256 complete and parseable；
- 256 prompt routes、single-tool exposures、verified executions 和 feedback
  result matches；
- harness vs 4B direct、 dual 9B direct 均 CI lower > 0、McNemar p < 0.05、
  至少 12 wins、0 losses；
- 两个 family 分别对 4B/9B non-regression；
- 通过也只允许另行预注册 fresh local replication，不能访问已观察 canary、
  complete benchmark 或 independent holdout。

## Boundaries

- config SHA：`4e5785b9b4fc9dd5b3a76a95b438c2f2c3d505a2fae6231b81cfbf53ebbd6ad9`；
- V2 report SHA：`cd20bd3f6abccf3e8b70f8ec6504150dc30665fbe477364608ec8b34366ab0cc`；
- canary rejection SHA：
  `6f0fcebabd0bfb8099ec34e6465362c1c884524605484aec47251068e9f5b056`；
- model generation：false；
- evaluation started：false；
- canary accessed：false；
- benchmark accessed：false；
- holdout accessed：false。

观察结果后禁止修改 cases、markers、schema、executor、prompt、regex、retry、
budgets、fallback、gate 或重跑。
