# Qwen3.5 Semantic Skill Applicability Scan v1

## 目的

在任何 real-task model generation 前，先冻结 unchanged semantic router 和
source-fact extractor，再扫描完整 GSM8K/MMLU/GPQA question columns。

Scanner 不读 answer、MMLU choices 或任何模型 output，也不公开 raw question 或
case ID。

## Frozen Rule

- router：`prompt_marker_to_single_semantic_skill_v1`；
- exact markers 与 parent semantic mechanism 完全一致；
- extractor：`exact_labeled_integer_fields_v1`；
- 必须先唯一 route，再完整提取所有 labeled integer fields；
- 任何缺字段、歧义或未命中都 direct-preserve，不暴露 tool。

## Surface

- GSM8K：1319；
- MMLU：14042；
- GPQA-Diamond：198；
- total：15559；
- complete case manifest 不含 expected/answer：
  `true`。

## Decision

- eligible rows >= 1：只允许另行预注册 exact real-task treatment；
- eligible rows = 0：关闭 unchanged semantic-skill real-task transfer；
- 观察 coverage 后禁止修改 markers、extractor、threshold、case selection、
  prompt、schema 或 executor；
- model generation、benchmark generation、canary rerun、holdout、training
  全部保持关闭。

## Boundary

- config SHA：`3cd70401fe00fcfe6664c7209438349a0f72882599a8f697d5784fc04a792c74`；
- replication report SHA：
  `da143f04a30775f73a147374844deb3b7b0534a0d2bd40a55cc9af9fc14335f0`；
- scan started：false；
- model generation started：false；
- benchmark generation started：false。
