---
name: tool-recovery
description: Handle tool calls, observations, schema failures, and bounded retries for compact agents. Use when a task has skill tag tool-use.
---

# Tool Recovery

1. Follow the tool schema exactly.
2. Treat every tool response as an observation, not a success assumption.
3. Preserve the failed call and error before changing arguments.
4. Retry only after changing the cause of the failure.
5. Stop when the tool error budget is exhausted.
6. Claim success only after a confirming observation.
