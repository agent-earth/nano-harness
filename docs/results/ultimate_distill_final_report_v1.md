# Ultimate Distill 最终全栈实验报告 v1

生成日期：2026-08-20

## 一句话结论

同一个 Qwen3.5-4B 底座，按 benchmark 走预先冻结的不同答题流程后，
在 **MMLU、GPQA-Diamond、MBPP 三个完整公开 benchmark** 上显著超过
同条件 Qwen3.5-9B。GSM8K 明显落后，必须保留为负结果。对 Qwen3.5-27B，
4B 在完整 MBPP 上没有达到预设的 -2 个百分点非劣门槛；在另一个 256 题
本地精确工具能力集上，4B harness 为 256/256，27B 直答为 63/256。

这不是“一个训练后的 4B checkpoint 全面超过 9B/27B”。公开 benchmark
的提升来自 **harness 路由和验证**，不是 SFT、DPO、RL 或 OPD。

## 先把几个词说清楚

- **同条件直接回答（matched direct）**：4B 和 9B 回答同一批题，提示词、
  输出长度、解码与评分器相同，只换模型。
- **Harness**：不改模型权重，只改变答题步骤。例如多次采样、保守投票、
  执行公开测试、失败后修复，以及不满足条件时退回原答案。
- **Verified**：让目标不可见的确定性代码或公开测试检查候选答案。只有检查
  通过才允许覆盖原答案。它是 harness 的一个迭代组件，不是训练方法。
- **完整 benchmark**：该 split 的全部样本都按预注册规则跑完。scan、
  import check、dry-run 和小样本开发集都不算正式分数。

## 完整公开 Benchmark

| Benchmark | 4B 使用的冻结流程 | 4B Harness | 9B 直接回答 | 分数差 | 95% CI | McNemar p | 结论 |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| mmlu | frozen_four_b_direct | 10273/14042 (73.16%) | 9066/14042 (64.56%) | +0.0860 | [+0.0786, +0.0936] | 1.24172e-112 | 通过 |
| gpqa_diamond | frozen_v5_conservative_choice_consensus | 85/198 (42.93%) | 69/198 (34.85%) | +0.0808 | [+0.0051, +0.1566] | 0.0479403 | 通过 |
| gsm8k | frozen_conditional_majority_v1 | 1220/1319 (92.49%) | 1243/1319 (94.24%) | -0.0174 | [-0.0288, -0.0061] | 0.0051524 | 失败 |
| mbpp | frozen_iterative_repair_v2 | 219/257 (85.21%) | 198/257 (77.04%) | +0.0817 | [+0.0389, +0.1284] | 0.00050826 | 通过 |

四项完整 benchmark 被放在同一个 Holm-Bonferroni 检验族中，familywise
alpha 为 0.05。MMLU、MBPP、GSM8K、GPQA-Diamond 的阈值依次为
0.0125、0.016667、0.025、0.05。四项差异都显著，但方向是三胜一负：
MMLU、GPQA-Diamond、MBPP 胜，GSM8K 负。

### 提升具体来自哪里

- 完整三任务直接回答基线共有 15,559 题：4B
  11,552 题正确，9B 10,375 题
  正确。直接回答只能在一个 benchmark 上形成显著胜利。
- **MMLU** 没增加模型调用，保留 4B 直接答案。4B 为 10,273/14,042，
  9B 为 9,066/14,042。9B 有 3,322 个严格格式解析失败，所以这里证明的是
  当前冻结输出合同下的完整任务表现，不应外推成裸模型知识能力差距。
- **GPQA-Diamond** 先保留 4B 的直接答案，再独立检查四个选项；只有两个
  独立检查和确认步骤同意同一个非原选项时才覆盖。4B 从 76/198 提高到
  85/198，9B 为 69/198。
- **GSM8K** 只在触发条件满足时做 5 次采样和多数判断，否则退回 4B 原答案。
  它把 4B 从 1,204/1,319 提到 1,220/1,319，但仍低于 9B 的
  1,243/1,319，所以不能算胜利。
- **MBPP** 先让 4B 写代码并运行公开测试。直接答案失败时再生成 5 个候选，
  按公开测试通过数选最好的一份；仍未全过才最多修复 3 轮。完整 test 共
  生成 340 个候选和 116 次修复，产生 38 次严格改善且没有使用隐藏参考解。

## MBPP 为什么可信

| 阶段 | 数据 split | 题数 | 4B Harness vs 9B | 分数差 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| development | train | 120 | 108/120 vs 97/120 | +0.0917 | method_supported |
| fresh_confirmation | full_validation_minus_sanitized_validation | 47 | 36/47 vs 31/47 | +0.1064 | directional_but_not_significant |
| independent_exact_replication | full_train_minus_sanitized_train | 254 | 211/254 vs 187/254 | +0.0945 | replicated |
| complete_sanitized_test | sanitized_test | 257 | 219/257 vs 198/257 | +0.0817 | complete_benchmark_win |

47 题确认集的方向是正的，但统计不显著，所以当时明确冻结为负证据，没有
直接打开 test。随后在另一个 254 题集合上用完全相同策略复现，达到显著性
门槛，才一次性运行 257 题 sanitized test。最终 4B harness 为 219/257，
9B 为 198/257；4B 直接回答只有 189/257。

## 与 27B 的比较

### 完整 MBPP：没有追平

4B harness 为 219/257，27B 直接回答为
226/257，差
-2.72%，95% CI
[-7.00%,
1.56%]。预注册要求置信区间下界
不低于 -2%，实际下界是
-7.00%，因此判定失败。

### 256 题本地精确工具能力集：明显超过

这套题只有四类可由确定性代码精确求解的问题。4B harness 为
256/256，27B 直接回答为
63/256，差
75.39%，95% CI
[69.92%,
80.47%]。四个题型都通过 -2%
非劣门槛，整体显著超过 27B。

边界必须保留：这是完整的本地合成能力集，不是公开 benchmark，不能用它
宣称“4B 全面超过 27B”。

### 27B 服务验证

- 模型通过 `BUCKET=ai-infra oniond download model Qwen3.5-27B` 下载。
- vLLM 0.19.1，BF16/FP16 权重，2 张 V100，tensor parallel 2，
  4,096 context，1 GiB 显式 KV cache，3/3 确定性 smoke 通过。
- GPTQ-Int4 在 V100 上输出连续 `!`，vLLM 也警告 4-bit `gptq_gemm`
  有数值问题；Marlin 不支持 compute capability 7.0。该量化服务被排除，
  没有拿来评分。

## 数据生成：生成了多少，实际训练了多少

| 数据版本 | Train rows | Dev rows | Train tokens | 数据门禁 |
| --- | ---: | ---: | ---: | --- |
| skill-sft-10k-10m-v2 | 15,888 | 400 | 11,425,166 | 通过 |
| orca-math-sft-v1 | 32,768 | 1,024 | 12,820,576 | 通过 |
| orca-math-preference-v1 | 512 | 192 | — | 通过 |
| paired-consistency-replication-v1 | 640 | 512 | 405,007 | 通过 |

最大的 skill 数据池确实达到了 **15,888 条 train / 11,425,166 train
tokens**。它由 55 个并行 shard 生成，每个 shard 都有 generator 和 critic
调用；全局去重删除了 11,872 条跨 shard 重复。

但这不等于已经用 11.43M tokens 完整训练一次。已完成的对应训练是：

- 长序列运行 smoke：从数据池取 10 条训练样本，4 steps，4/5 → 4/5；
- bounded-dose SFT：取 80 条，20 steps，17/20 → 17/20；
- reasoning-preservation SFT：取 80 条，20 steps，16/20 → 16/20。

三次都证明训练、保存、重载链路可运行，但没有证明质量提升。完整 10M-token
训练没有启动，因为小规模门禁没有通过，继续扩大只会放大成本和风险。

Orca Math 数据池有 32,768 条 train、1,024 条 dev、12,820,576 train
tokens，并通过与固定 GSM8K/MMLU/GPQA 语料零精确/近似重叠检查。实际
Orca SFT smoke 只使用 160 条训练样本、40 steps；结果从 100/192 降到
58/192，因此停止。

## 训练消融

| 实验 | 实际训练样本/对 | Steps | 评测题数 | 前 → 后 | 决策 |
| --- | ---: | ---: | ---: | --- | --- |
| 专项算术 SFT | 512 | 128 | 96 | 0 → 2 | rejected_nonsignificant |
| 10M 数据运行 smoke | 10 | 4 | 5 | 4 → 4 | runtime_smoke_only_no_quality_gain |
| 10M 数据小剂量 SFT | 80 | 20 | 20 | 17 → 17 | rejected_no_quality_gain |
| 推理样本加权 SFT | 80 | 20 | 20 | 16 → 16 | rejected_no_reasoning_gain |
| Orca Math SFT | 160 | 40 | 192 | 100 → 58 | rejected_regression |
| 整段 DPO | 32 | 32 | 192 | 91 → 91 | rejected_no_effect |
| 只训 FINAL 后缀的 DPO | 32 | 32 | 192 | 83 → 83 | rejected_no_correctness_change |
| 过程到答案一致性训练 | 见配置 | 40 | 80 | 51 → 55 | directional_only_not_significant |
| RL / OPD 实现 smoke | 2 | 2 | — | 实现前未运行 → 两种方法均完成 2-step smoke | implementation_only |
| RL / OPD 质量检查 | 见配置 | — | 96 | Base 3 → RL 3 / OPD 3 | rejected_no_quality_gain |
| 三分类 router SFT | 见配置 | 40 | 192 | 112 → 192 | local_router_behavior_admitted_not_benchmark |
| 八类负例 router SFT | 见配置 | 40 | 1536 | 1127 → 1536 | local_router_behavior_admitted_serving_remap_required |

结论很直接：

- 标准 SFT 有过明显回归，也有 0/96 → 2/96 的微小正向结果，但后者
  McNemar p=0.5，不显著。
- 两个 DPO 实验分别是 91/192 → 91/192、83/192 → 83/192。第二个目标
  更聚焦，改了 4 个输出，但没有改对任何一题。
- RL 和 OPD 各跑了 2 个 optimizer step，证明实现能训练、adapter 能重载、
  logits 会变化；随后在 96 题上两者都是 3/96，与 base 相同。
- 一致性目标在 80 题上把 verifier 分数从 51 提到 55、exact 从 32 提到
  33，首次修好一个 process-to-final pair；p=0.21875，只能算方向性机制
  证据。
- Router SFT 在本地合成分类上从 112/192 提到 192/192，扩展负例版本从
  1,127/1,536 提到 1,536/1,536。这证明 router 行为可学，但不等于真实
  benchmark 提升，且后者还要求 serving namespace remap。
- `SFT+RL` 没有运行。原因是没有 SFT 或 RL 候选先通过质量门禁，组合两个
  未通过的组件没有可辩护的实验依据。

## Agent Benchmark 可跑性

SkillBench、ClawBench、WildClawBench、Terminal-Bench 2、SWE-bench
Lite 的 5/5 本地检查通过：包括真实 skill scan、319/319 ClawBench
定义、60/60 WildClawBench 任务、89/89 Terminal-Bench 2 manifest，以及
300 行 SWE-bench Lite parquet。

这些都是可运行性检查，不是分数。本机缺少官方执行需要的 container mount
namespace 权限；Docker、rootless Podman 和 Buildah 路径都在运行测试前
失败。因此本报告的 agent benchmark 正式得分仍为 0 项。

## 最终消融判断

| 层 | 结果 |
| --- | --- |
| 4B direct baseline | 完整三任务只显著赢 1 项 |
| Harness-only | 完整四项中赢 3 项；这是主结果 |
| Data-only | 规模、去重、来源和泄漏检查通过；不代表模型提升 |
| SFT | 局部正向、无变化和显著回归都出现；无公开 benchmark 增益 |
| DPO | 两次均无正确率变化 |
| RL | 实现通过，96 题质量无提升 |
| OPD | 实现通过，96 题质量无提升 |
| SFT+RL | 未执行；前置组件未通过质量门禁 |
| Verifier / tool | 在 MBPP 和 256 题精确工具集上有效 |

因此，当前能复现的三项公开 benchmark 提升应归因于 **harness 路由与
验证**，不能归因于训练。

## 证明了什么，没证明什么

**已经证明**

1. 一个 Qwen3.5-4B 底座配合预先冻结的 benchmark-specific harness，
   在完整 MMLU、GPQA-Diamond、MBPP 上显著超过 matched 9B。
2. 目标不可见的确定性执行器在适用题型上能让 4B 大幅超过 27B 直答。
3. 数据管线能生成超过 10M tokens 的可追踪、可去重、可做泄漏检查的数据。
4. 当前训练框架可稳定完成 SFT、DPO、RL、OPD smoke 并重载 adapter。

**没有证明**

1. 一个 fine-tuned 4B checkpoint 同时赢下三个 benchmark。
2. 4B 在完整 MBPP 上追平 27B；实际结果是 219/257 对 226/257。
3. SFT、DPO、RL、OPD 或 SFT+RL 带来了稳定 benchmark 提升。
4. 本机取得任何正式 agent benchmark 分数。
5. 4B 在本地精确工具能力集以外广泛超过 27B。

## 复现与证据

- 主仓库：`https://github.com/steven-kid/nano-harness.git`
- 分支：`fullstack/campaign-v1`
- 报告生成基线提交：`a4823fdba723b9446dff47cb3cf6dd9d1572dc65`
- 数据仓库提交：`ab8c7a6e74cb84f0f935bb398b8fe82536946340`
- 训练仓库提交：`ea8c6a07d945db1233a03e899f8973c6257ea13d`
- 主结论 JSON SHA256：
  `34c2011fde8067878ef17e18a79327a55fca00d70ab3ea5282a4a91ecf16c06d`
- 27B 工具能力 JSON SHA256：
  `34f56147c47e793024a8f2e7cad9ddcb9bfbcb54ebbbffd82796f2ca3c026d18`
- 27B MBPP 负结果 JSON SHA256：
  `1ae47518b33518fb29c8269bb5d62aa1f0eaf8f67fc306214f1209eea8e19d7b`

重新生成：

```bash
cd ultimate-distill-workspace/worktrees/nano-harness-fullstack-traex-03
PYTHONPATH=. ../../.venv/bin/python scripts/render_ultimate_distill_final_report_v1.py
```

完整回归：

```bash
PYTHONPATH=. ../../.venv/bin/python -m unittest discover -s tests -v
```

原始生成结果、模型权重和日志保留在本地 ignored 路径；GitHub 只提交配置、
测试、预注册收据和 public-safe 报告。
