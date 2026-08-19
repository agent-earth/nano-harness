---
name: tool-recovery-v2
description: Candidate tool-recovery policy that classifies failures before bounded retries. Use only in frozen synthetic skill evaluation until promotion.
---

# Tool Recovery V2

1. Follow the tool schema exactly.
2. Treat every tool response as an observation.
3. Classify the failure before retrying: schema, missing prerequisite,
   transient provider error, permission boundary, or unsupported action.
4. Repair only the classified cause. Do not repeat an unchanged failing call.
5. Preserve the error and changed argument receipt.
6. Stop on permission, unsupported action, or exhausted error budget.
7. Claim success only after a confirming observation.
