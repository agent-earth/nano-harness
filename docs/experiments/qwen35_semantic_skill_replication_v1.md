# Qwen3.5 Typed Semantic Skill Replication v1

## 唯一变化

这次保持 parent semantic-skill mechanism 完全不变，只换证据分布：

- 新场景：compact display / kiosk forecast；
- 新数值范围：中小整数，不再使用 parent 的超大 offset；
- 新 case seed 和 case IDs。

Router markers、skill prompts、tool schemas、semantic executor、source validator、
one retry、direct fallback、result feedback、feedback equality、models、services、
temperature、budgets 和统计 gate 都不变。

## Freshness

- 256 cases，2 families × 128；
- case contract SHA：
  `e863bb74a2e58bb8287f004603c7e450c18743a597ba72ad1cd6fa4989ca1e74`；
- parent case ID overlap：0；
- parent prompt overlap：0；
- parent source-fact overlap：0；
- prior choice/tool prompt overlap：0；
- complete GSM8K/MMLU/GPQA prompt overlap：0；
- benchmark/canary/holdout outputs 或 rows：0。

## Gate

- 256/256 complete and parseable；
- 256 prompt routes、single-tool exposures、verified executions、feedback matches；
- 0 retry/fallback/contract failure；
- harness vs 4B 和 vs 9B 均 CI lower > 0、McNemar p < 0.05、
  至少 12 wins、0 losses；
- every-family non-regression。

通过也只允许另行预注册 real-task transfer；不允许重跑已观察 canary，不允许
直接生成 complete benchmark，不开放 independent holdout 或 training。

## Boundaries

- config SHA：`90774e3be32504637ee5cca27c45a8c577c7b9803e9c46c7bb68381b81a79501`；
- parent report SHA：`fe53a512cbf0b6ada65ed3ae27c5f3dc90165e367cfecdb58307dd030d017d5f`；
- model generation：false；
- evaluation started：false；
- canary / benchmark / holdout accessed：false；
- training started：false。

观察结果后禁止修改模板、数值范围、markers、schema、executor、prompt、
regex、retry、budget、fallback、gate 或重跑。
