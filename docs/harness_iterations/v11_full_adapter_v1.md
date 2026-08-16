# V11 Full Matched Adapter v1

## Hypothesis

V11 passes its local family gate and the sealed 40-case regression canary. The
full test checks whether the unchanged adapter preserves base Qwen3.5-4B
behavior and improves over the frozen Qwen3.5-9B baseline on the prior matched
211-case suite.

## Frozen Contract

Reuse `qwen35_v6_matched_adapter_v1.yaml` without changes:

- 96 GSM8K cases;
- 96 MMLU cases;
- 19 GPQA-Diamond cases;
- identical case IDs, prompts, system prompts, task budgets, scorers,
  temperature, and `enable_thinking: false`;
- frozen base 4B and 9B raw results from the three-task replication;
- exact v11 adapter authorized by the passing local and canary receipts.

The candidate model is `qwen3.5-4b-targeted-v11` at the local vLLM endpoint.
Namespace conversion changes only module keys and preserves all 224 tensor
contents.

## Gate

Report candidate versus base 4B and 9B separately, including task and aggregate
scores, paired bootstrap intervals, exact McNemar tests, parse failures,
truncations, API errors, latency, token use, discordant case IDs, and raw
SHA256 identities.

The adapter may continue only if:

- no task point estimate is below base 4B;
- aggregate micro and macro are not below base 4B;
- candidate versus base 4B micro delta is not significantly negative;
- candidate macro exceeds 9B;
- no task point estimate is below 9B;
- no API, case, prompt, budget, scorer, or serving-parity defect occurs.

Passing this matched suite still does not authorize merge, scale-up, or RL. It
only permits a separately reasoned next experiment.

## Frozen Identity

- suite manifest SHA256:
  `08c71cae463bd3b0a0031e95d6339136d0c445beecaac631c4f5843e0b14d4c1`;
- base 4B raw SHA256:
  `c59383d3fd3d6087025d6e1ff649979d9d5a9e8dc73b5429a4f8e9fa41b6b8c7`;
- 9B raw SHA256:
  `ffae93774d51b87a2e29258d170a84f8b165f996e2e78eedd102271dfc260044`;
- v11 adapter tree SHA256:
  `87248908918b06c2d28ff68efd4f0b1ff92ca8bf8b7588e1c7e81a85eb7da852`;
- canary candidate raw SHA256:
  `85d993405f6f7813e4dfaba2b813719fb3b0abda6558f3827aec463ed591a4cb`;
- canary public receipt SHA256:
  `5622f50f7207657d5923b9309943cb8572c3e16847210ec1368c15baf5ff4345`.

## Reproduction

```bash
PYTHONPATH=. ../.venv/bin/python -m nano_harness.cli baseline \
  --manifest configs/harness/qwen35_v6_matched_adapter_v1.yaml \
  --dataset-root ../../datasets \
  --model qwen3.5-4b-targeted-v11 \
  --base-url http://127.0.0.1:8003/v1 \
  --output \
    results/harness/qwen35-v11-full-matched-adapter-v1/candidate/cases.jsonl
```
