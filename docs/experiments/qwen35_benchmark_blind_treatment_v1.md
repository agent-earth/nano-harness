# Qwen3.5 Benchmark-Blind Treatment v1

## 这次具体做了什么

这次只冻结下一轮 4B treatment 的实验设计，没有跑新的训练、canary 或
完整 benchmark。

- treatment：`qwen35-benchmark-blind-treatment-v1`
- config SHA256：`b90f9ab9e1a2328c04f7e005c56373978344caae5763b7f6a25923eaac6a6a7b`
- peer 预注册 commit：`6b6ac50f706ab29edfce267a4d286ed408628012`
- peer replication result 是否存在：`false`
- 当前是否允许跑 canary：`false`
- 当前是否允许跑完整 treatment benchmark：
  `false`

当前结论：Peer consistency replication result is absent or fails at least one frozen admission gate.

## 为什么不能直接跑完整 benchmark

完整 direct 基线已经证明 4B 只通过 1/3 个 benchmark gate。下一次实验必须
先用与 benchmark 无关的训练信号证明一致性 adapter 稳定有效，再用冻结的
211-case canary 检查是否破坏 GSM8K、MMLU、GPQA。peer replication 缺少任一
显著性、reload、finite 或 JSON 非回归 gate，三种 treatment 全部禁止生成。

## 三个固定消融臂

| Arm | 唯一变化 | 实验内容 |
| --- | --- | --- |
| `adapter_only` | `adapter` | The unique adapter emitted by the accepted peer paired-consistency replication; direct harness unchanged. |
| `arbiter_only` | `gpqa_harness` | Base 4B with direct GSM8K/MMLU and the frozen option-evidence conservative arbiter on GPQA only. |
| `adapter_plus_arbiter` | `adapter_and_gpqa_harness` | The unique accepted consistency adapter composed with the unchanged GPQA-only arbiter. |

`arbiter_only` 不是已晋级的 harness。它在 GPQA dev8 上 2 次覆盖均修正答案，
但在后续 72-case holdout5 上 1 赢 1 输，最终 gate 未通过。因此它只作为固定
ablation 继续验证，不能单独写成“verified 提升”。

## 已冻结的历史证据

- `complete_direct_baseline`：`/decision/complete_benchmarks_significantly_won` = `1`, `/decision/project_goal_gate_passed` = `false`, `/decision/rl_allowed` = `false`。
- `gpqa_arbiter_dev8_direction`：`/decision/accepted` = `true`, `/override_analysis/override_count` = `2`, `/override_analysis/neutral_overrides` = `0`。
- `gpqa_arbiter_holdout5_rejection`：`/decision/accepted` = `false`, `/override_analysis/gpqa_override_count` = `2`, `/decision/per_benchmark_non_regression_vs_9b` = `false`。

这些值只用于说明为什么选择“一致性 adapter + GPQA arbiter”做机制消融，
不能把完整 benchmark 的错题、答案或模型输出送进训练、reward、verifier 或
数据生成。

## Canary 准入

三个 arm 都必须独立跑完，不因前一个结果好坏而跳过：

- 固定 211 cases；case、dataset、prompt、parser、scorer 完全一致；
- overall 至少 `164/211`；
- GSM8K 至少 `90/96`；
- MMLU 至少 `67/96`；
- GPQA 至少 `6/19`；
- API error 为 0，parse failure 最多
  `2`；
- 相对 base 4B 的 candidate-only wins 必须多于 base-only wins。

任一 arm 失败就保留为负证据，不能在这 211 rows 上改 prompt、budget、route、
adapter weight 或训练参数修复。

## 完整 benchmark 准入

只有独立通过 canary 的 arm 才能跑完整 GSM8K、MMLU、GPQA。每个 admitted
arm 都要执行，不能看第一个完整结果后只挑最好看的 arm。最终 4B 超过 9B 的
正式 gate 仍是：

- 每个 benchmark candidate accuracy 更高；
- paired bootstrap 95% CI 下界 > 0；
- exact McNemar `p < 0.05`；
- candidate-only wins > 9B-only wins；
- 相对 direct 4B 每个 benchmark 不回退；
- 三个完整 benchmark 全部通过；
- strict score 是唯一正式分数，loose-format 只做非评分诊断。

## 禁止事项

观察任何 treatment 输出后禁止：

- `training_data_change`
- `objective_change`
- `loss_weight_change`
- `teacher_detach_change`
- `learning_rate_change`
- `seed_change`
- `step_change`
- `lora_scope_change`
- `adapter_checkpoint_choice`
- `adapter_weight_change`
- `prompt_change`
- `parser_change`
- `scorer_change`
- `token_budget_change`
- `benchmark_route_change`
- `arbiter_threshold_change`
- `case_selection_change`

独立 holdout 继续密封。RL/OPD 也不会因为这份预注册自动开放。

## 下一步

Wait for and consume the peer-owned unique paired-consistency replication result; do not run treatment canary or complete benchmarks.
