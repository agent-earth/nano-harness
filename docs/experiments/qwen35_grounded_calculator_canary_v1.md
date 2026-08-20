# Qwen3.5 Grounded Calculator Canary v1

## 具体实验

这次只预注册，不生成新的 canary output。

冻结的 211 cases 包含 GSM8K 96、MMLU 96、GPQA-Diamond 19。4B direct
原结果是 163/211，只有 2 行无法解析；9B direct 是 151/211。

候选 harness 不重新生成 209 行：

- MMLU、GPQA 全部复用原 4B direct output；
- GSM8K 中已有 `prediction` 的行也复用原 output；
- 只有 `benchmark=gsm8k && status=completed && prediction=None` 才进入
  calculator recovery；
- 动态资格条件不读取 expected、score 或 case correctness，也没有 case-ID
  allowlist。

## Grounded Calculator

Recovery 让同一个 4B 输出一条 `CALC: <expression>`。安全执行器：

- 只允许 `+ - * /`、括号和一元正负号；
- 每个数字字面量都必须来自原题面，允许复用题面数字；
- 使用 `Fraction` 精确计算，不使用 `eval`；
- AST 最多 64 nodes，表达式最多
  160 chars，结果必须是整数；
- 失败时原样回退 direct；
- 成功后把 verified result 回传同一个 4B，只允许 32-token `FINAL`。

Plan budget 是 96，one retry；最大实测 plan input
190 tokens，低于 4096 context。

## 原 Gate 不变

- overall 至少 164/211；
- GSM8K 至少 90/96；
- MMLU 至少 67/96；
- GPQA 至少 6/19；
- parse failures 最多 2，API errors 为 0；
- candidate-only wins 必须多于 base-only wins；
- 所有非 eligible 行的评分字段必须与 frozen direct 完全相同。

## 边界

- config SHA256：`3c4049a89ff34895989a82450d78b1d74c3e5889ac9d24307e2b93f2dfe230a2`；
- case manifest SHA256：`eafbe4d42487a225322dd3b3bdc1d805c065fb15f0f8b968e65ccf747f96976f`；
- V2 public report SHA256：`cd20bd3f6abccf3e8b70f8ec6504150dc30665fbe477364608ec8b34366ab0cc`；
- eligible set 只记录 SHA256，不作为运行 allowlist：
  `755fd35aac339fb701bc081aeb421d184e3c66c1dbdb33f6cbb28b17dab302e5`；
- candidate generation：false；
- complete benchmark：关闭；
- independent holdout：密封；
- training eligible rows：0。

观察结果后禁止修改 eligibility、prompt、regex、AST grammar、grounding、
retry、budget、fallback、parser、scorer、gate 或重跑。
