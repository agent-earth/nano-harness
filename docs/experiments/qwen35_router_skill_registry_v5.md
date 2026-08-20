# Qwen3.5 Router Skill Registry v5

V4 improved the candidate to 142/160 but its shared eight-schema selector
failed on 20 cases. V5 uses target-blind applicability predicates and exposes
exactly one schema after a unique match.

- fresh cases: 160, ten families x 16;
- C registry unique matches: 128/128;
- A/B false registry matches: 0/32;
- overlap with training, V1-V4, prior surfaces, GSM8K, MMLU, GPQA: zero;
- config SHA: `f8544c6337948be87e8a34721bc10906fc724712d8b0ffed24381ef13a59ee91`;
- case contract SHA: `24211fb1eb73f14a23d80ee365c496d6fab586fcc59f2ca2ddbccc92377371bb`;
- model generation started: false.

Passing permits only separately pre-registered benchmark treatment generation.
V1-V5 cannot be rerun after observation.
