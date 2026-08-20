# Qwen3.5 Router Serving Parity v1

## Question

The HF reload scores 192/192, but the first vLLM integration routes all 64
unsupported rows to A. This diagnosis separates data transfer from deployment:

- base 4B;
- original PEFT namespace `model.layers.*`;
- content-identical remapped namespace `language_model.model.layers.*`.

## Frozen Evidence

- rows: 192 observed SFT validation rows, A/B/C = 64/64/64;
- fresh integration, benchmark, canary, and holdout rows loaded: 0;
- original/remapped tensors: 224, identical dtype/shape/content hashes;
- tokenizer vocab, special tokens, chat template, every prompt ID, and every
  target ID are semantically equal.

## Gates

- all three arms complete and parseable;
- remapped output is exact 192/192 and byte-equal to HF output 192/192;
- each remapped label is 64/64;
- original output differs from HF on at least one row;
- remapped exact is strictly greater than original exact.

Passing identifies the namespace mismatch as the serving root cause. It permits
only a separately pre-registered **new history-disjoint integration v2**. It
does not permit rerunning observed integration v1, a real question scan,
benchmark/canary/holdout access, training, or RL.

## Identity

- config SHA: `025f6208541ae0e88a4853eceb04d2f185293daccb650890d43d297e86b0c7b6`;
- case contract SHA: `b87e3fd102b3cfb765e79add32124529fce68a5e0fa5e33680cd32534b5eb2c2`;
- original adapter SHA: `48d72666cf269f64825791801c71aad5ae1d9e97cbc70574c216b12974e46e63`;
- remapped adapter SHA: `fbaa39dcb3fcf34e9aab280308cb5a5416094c1968e4ac3a69cd739a806ecc49`;
- model generation started: false.
