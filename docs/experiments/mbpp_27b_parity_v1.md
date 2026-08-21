# MBPP 27B Parity v1

This freezes one complete 257-case MBPP parity comparison before any 27B
benchmark generation.

- 4B candidate: reuse only, raw SHA
  `86c72abd02adb88cc2ae02bf74893068c0c4842c1dc214dbcc85a6306108d944`;
- 27B: validated BF16 TP=2 vLLM service, 4096-token context;
- case IDs SHA: `4b7e1f74447041a3f9ad4c02a27c157fd1ef98beb2c770796f4bfd8e630863d8`;
- config SHA: `dbeccb9afc6acc4100c853a4e8bc52385e9fbc38d52204486e15296c286549e1`;
- shards: `[129, 128]`;
- noninferiority margin: 2 percentage points.

Parity passes only when the paired-bootstrap 95% lower confidence bound for
4B candidate minus 27B is at least -0.02. The run is one-shot and its rows or
outputs may not enter training, reward, verifier fitting, or tuning.
