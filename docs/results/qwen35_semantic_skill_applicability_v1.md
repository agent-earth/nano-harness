# Qwen3.5 Semantic Skill Applicability v1 Result

## 结论

**Unchanged exact-marker semantic transfer 关闭。**

- scanned questions：15559；
- route missing：15559；
- eligible rows：0；
- model generation：0；
- answer columns / choices / model outputs：均未读取。

Parent mechanism 的 exact markers 和 exact labeled-field extractor 在完整
GSM8K、MMLU、GPQA question surface 上没有任何可执行覆盖。根据预注册规则，
现在不能基于已观察 prompts 扩 marker 或修改 extractor，也不能生成 benchmark
outputs。

## Coverage

```json
{
  "cases": 15559,
  "cases_by_benchmark": {
    "gpqa_diamond": 198,
    "gsm8k": 1319,
    "mmlu": 14042
  },
  "eligible_by_benchmark": {},
  "eligible_by_family": {},
  "eligible_rows": 0,
  "extraction_failures": {
    "route_missing": 15559
  },
  "route_counts": {
    "route_missing": 15559
  }
}
```

## Boundary

```json
{
  "answer_columns_loaded": false,
  "case_correctness_used": false,
  "case_ids_published": false,
  "choices_column_loaded": false,
  "expected_answers_used": false,
  "model_generation_started": false,
  "model_outputs_loaded": false,
  "question_column_only": true,
  "raw_questions_published": false
}
```

只发布 aggregate counts 与 hash-set identities；不发布 raw questions 或 case
IDs。

## Evidence

- prereg commit：`0f3b1ed833ad64bf9c09c2f4468fa45262d3f53f`；
- config SHA：`3cd70401fe00fcfe6664c7209438349a0f72882599a8f697d5784fc04a792c74`；
- prereg SHA：`8b754f42b0704bb5425b4ab936382ffded8774f19fa70f69f01f71e5437c277c`；
- raw result SHA：`68b197992a5b8c427b6b9cb319d30d781e5d18eb029b90e81b8571cd5ad5c2a4`；
- question hash-set SHA：
  `86965ce6f6835ae455fe4c5ed56e37e61459e71312bee5bd99226e9c24bf97cc`；
- empty eligible-set SHA：
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

## 下一步

只允许在 fresh non-benchmark surface 设计 model-selected semantic router，
用 enum-constrained route selection 后再单 skill exposure；必须先通过 fresh
local route/execute/fallback/zero-loss gate，再另行预注册 real-task scan 或
treatment。当前 benchmark generation、canary rerun、holdout 和 training
继续关闭。
