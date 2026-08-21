# Qwen3.5-27B BF16 TP=2 Serving v1

The two-GPU BF16 service passed 3/3 deterministic smoke probes at 1024 context. The GPTQ-Int4 service is rejected because vLLM 0.19.1 warns that its 4-bit GPTQ GEMM is buggy and the observed outputs degenerated to punctuation; Marlin has no supported quantization type on the V100 compute capability 7.0 GPUs.

This is serving evidence only and establishes no benchmark score.
