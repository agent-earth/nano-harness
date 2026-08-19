# Qwen3.5 Complete Direct Baseline v1

## 结果

完整数据共 15,559 个 matched cases。两臂 case、prompt、system、parser、
scorer 和 generation budget 完全一致，均无 API error。

| Benchmark | Qwen3.5-4B | Qwen3.5-9B | 4B - 9B | Paired 95% CI | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 1203/1319 (0.9121) | 1240/1319 (0.9401) | -0.0281 | [-0.0409, -0.0152] | 3.762e-05 |
| mmlu | 10273/14042 (0.7316) | 9066/14042 (0.6456) | +0.0860 | [+0.0786, +0.0935] | 1.242e-112 |
| gpqa_diamond | 76/198 (0.3838) | 69/198 (0.3485) | +0.0354 | [-0.0404, +0.1111] | 0.4188 |

全体 micro：4B 11552/15559
(0.7425)，9B
10375/15559
(0.6668)，delta +0.0756，
95% CI [+0.0688, +0.0824]，
McNemar `p=3.299e-102`。

## 证明了什么

- MMLU strict scorer：4B 显著高于 9B；
- GSM8K：4B 显著低于 9B；
- GPQA-Diamond：4B point estimate 更高，但不显著；
- 因此“三个完整 benchmark 都显著超过 9B”的目标只完成
  **1/3**，不能宣称目标达成。

## 格式诊断

strict scorer 不变。下面只解释失败来源：

- 9B MMLU 有
  3266
  个 `FINAL <letter>` 缺冒号输出；其中
  1760 个字母与
  reference 一致。若只做非评分 colon-normalized 诊断，9B MMLU 为
  0.7710，
  高于 4B strict
  0.7316。
- 9B GPQA 同类诊断为
  0.4091，
  高于 4B strict
  0.3838。
- 这说明官方 strict MMLU 优势主要是格式遵循，不应解释为稳定语义优势。
  官方 strict 分数与所有 paired 统计不做任何改写。

## 下一步

保持 benchmark rows/outputs 禁止训练，也不在完整结果上搜索 prompt、budget、
parser 或 scorer。下一阶段应消费独立开发面和 peer mechanism evidence，预注册
一个不读取 benchmark 内容的 4B treatment：

1. 优先修复 GSM8K 语义执行，因为这是完整基线中唯一显著负项；
2. GPQA 需要可迁移的 verifier/skill 改进；
3. MMLU 必须保留 strict score，同时把格式诊断作为稳健性保护项；
4. treatment 先过 local/canary，再一次性跑相同完整 case set。

## Evidence

- generation revision：`543e4b8c4164b50b07a1aa87c16210a158a9f34d`
- analysis revision：`398f7160f70a3b76064f4bc94cd1fb45d68945a0`
- 4B raw SHA：`b6e35d968dc3ff3311cf991b186e7a4f2a0ff8fe9989b7507f39b75b7aa6aa54`
- 9B raw SHA：`a22cade1ea367e6cceddff4e25e88d34360adc005feeed316464b561c151b988`
- comparison SHA：`85841b4641606e3188a055f99269f0c5748b7e58e6724a5e00916e0c4162c5eb`
- case contract SHA：`858656f58decf8bbc23c70101dabcffc6ef12e049771e043575927743c6cfd10`
- 服务启动 receipt SHA：`d4157f935bcaacc4ca227b8f1ea4d500b3291f437f9898d6379a038bae212c41`

raw outputs 和服务日志保持 ignored，不进入 Git。
