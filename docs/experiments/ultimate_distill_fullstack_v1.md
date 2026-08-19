# Ultimate Distill Full-Stack Campaign v1

## 这次做了什么

这次只做全栈现状审计和下一轮实验预注册，没有启动模型推理、训练、
benchmark 跑分、RL 或 OPD。

- campaign：`ultimate-distill-fullstack-v1`
- base revision：`b538464`
- config SHA256：`dd756f4ed3db5b7fe18b5f80ef0d29d383fb4a9bcecb598ff0258b6b89d96189`
- 所有 14 个依赖文件身份已重算并通过

## 当前可以直接做什么

- 本机有两张 32GB V100；Qwen3.5-4B 和 Qwen3.5-9B 权重完整且身份通过。
- 已冻结三个完整 benchmark：
  - gsm8k: 1,319 rows, numeric_exact
  - mmlu: 14,042 rows, choice_exact
  - gpqa_diamond: 198 rows, choice_exact
- 下一步先跑三个完整 benchmark 的 matched 4B/9B direct baseline。
- skill harness、普通 SFT 和 paired consistency 已有版本化实现。

## 当前还不能做什么

- Qwen3.5-27B-FP8 未安装。`oniond` 可见该模型，但预注册时只剩
  39 GiB 磁盘；未获得单独清理决策前不删除现有模型、数据或 evidence。
- RL 和 OPD 目前没有版本化实现或 smoke receipt，因此继续关闭。
- SWE-bench Lite、ClawBench、WildClawBench、Terminal-Bench 2 等当前只有
  scan/parse/dry-run 证据；本机缺少可用容器 mount namespace，不能把 scan
  写成正式模型分数。

## 已复核的历史结论

- `small-matched-direct`：Small matched baseline only; no 4B superiority.
- `three-task-directional`：Directional three-task replication; aggregate significance gate failed.
- `large-direct-confirmation`：MMLU-only significant direct evidence; GSM8K regresses and aggregate significance fails.
- `verified-executor-mechanism`：Significant generic mechanism evidence only; not benchmark superiority.
- `paired-consistency-direction`：Directionally positive local mechanism; replication must pass before benchmark use.

skill 自进化 synthetic contract 复算结果：

- parent：4/6
- candidate：6/6
- promoted：`true`
- 这只证明 frozen synthetic skill contract 改进，不是 benchmark 提升。

## 候选阶梯

1. `matched-direct-complete-baselines`：Run direct Qwen3.5-4B and Qwen3.5-9B over every row of GSM8K test, MMLU no-train test, and GPQA-Diamond. 停止条件：Stop if any model, case, prompt, scorer, parser, or raw-output identity differs across arms.
2. `skill-harness-complete`：Run the frozen skill-routed and verified-execution harness over the same complete case sets. 停止条件：Reject on any complete-benchmark regression versus direct 4B or any identity mismatch; do not tune on observed full results.
3. `sft-complete`：Evaluate exactly one frozen Qwen3.5-4B SFT/consistency adapter on the same complete case sets. 停止条件：Reject the adapter if local replication, reload, or any complete benchmark gate fails.
4. `sft-plus-harness-complete`：Compose the admitted adapter and harness without changing either component and evaluate the same complete case sets. 停止条件：Reject if the combined arm regresses any benchmark relative to the stronger admitted component.
5. `rl-sanity-and-complete`：Run a verifier-guided RL sanity check and, only if admitted, one frozen complete-benchmark candidate. 停止条件：Keep RL closed until admission; after admission run once and reject on local or complete-benchmark non-regression failure.
6. `opd-sanity-and-complete`：Run one on-policy distillation sanity check and, only if admitted, one frozen complete-benchmark candidate. 停止条件：Keep OPD closed until admission; after admission run once and reject on local or complete-benchmark non-regression failure.
7. `twenty-seven-b-parity`：Evaluate the admitted 4B candidate against matched Qwen3.5-27B-FP8 on complete GSM8K and MMLU. 停止条件：No 27B parity claim is allowed without matched case-level outputs and the pre-registered non-inferiority confidence interval.

RL/OPD 不会因为出现在路线图里就自动获准。必须先补实现、污染审计、
finite smoke、reload、固定 config 和 no-post-hoc-search gate。

## 最终验收

- 在完整 GSM8K、MMLU、GPQA-Diamond 上分别与 matched 9B 做 paired 比较。
- 每个 benchmark 都要求 candidate accuracy 更高、bootstrap 95% CI 下界大于
  0、exact McNemar `p<0.05`、case/prompt/parser/scorer 完全一致、零 API error。
- 至少 3 个完整 benchmark 同时通过，才允许声称“4B 显著超过 9B”。
- 27B parity 只在完整 GSM8K 和 MMLU 上判断；预注册 non-inferiority margin
  为 0.02，两项都要通过。

## 审计结果

- all_artifact_identities_verified: `true`
- four_b_and_nine_b_ready: `true`
- twenty_seven_b_missing_is_explicit: `true`
- three_complete_benchmarks_verified: `true`
- formal_scans_not_reported_as_scores: `true`
- skill_evolution_reproduced: `true`
- peer_dependencies_verified: `true`
- rl_and_opd_fail_closed: `true`
- no_model_or_training_execution: `true`

## 执行边界

```json
{
  "benchmark_scoring_started": false,
  "model_generation_started": false,
  "opd_started": false,
  "rl_started": false,
  "this_commit_only_audits_and_preregisters": true,
  "training_started": false
}
```

下一可执行切片：Generate the frozen all-row case manifest and execute matched 4B/9B direct baselines on GSM8K, MMLU, and GPQA-Diamond without consuming benchmark outputs as training data.
