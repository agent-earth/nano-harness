# Qwen3.5 Router Serving Parity v2

## Question

The new negative-diversity adapter reaches 1,536/1,536 under an independent HF
reload. V1 already proved that Qwen3.5 vLLM requires a content-identical
namespace remap. This experiment asks only whether the new remapped adapter
reproduces all frozen HF outputs under vLLM.

## Frozen Evidence

- rows: 1,536 observed SFT validation rows, A/B/C = 512/512/512;
- C subtypes: 8 x 64;
- fresh integration, benchmark, canary, and holdout rows loaded: 0;
- remap: 224 tensors with identical dtype, shape, and content hashes;
- tokenizer vocab, special tokens, chat template, prompt IDs, and target IDs:
  equal.

## Gates

- all 1,536 outputs complete and parseable;
- all 1,536 outputs exactly match both target and HF output;
- every label is 512/512;
- every C subtype is 64/64.

Passing permits only a separately pre-registered, new history-disjoint
integration v3. It does not permit rerunning integration v1/v2, benchmark,
canary, holdout, training, or RL.

## Identity

- config SHA: `10f351e9e7209695a5a66f79b18a7265a7da3c66410be21fef584f0474e28b3b`;
- case contract SHA: `9b295456d56c7cb7d6acf2ecf2666ab50ad1fe564852080288ab31e9a36422d0`;
- original adapter SHA: `c40b6ab8811f162b60e6e290c7a9defbc0c013b239282bfbe8313eb986524c04`;
- remapped adapter SHA: `cea357d281ed100437268e213564fc5a5c00e6024b0c7a4be207cc686453e3f9`;
- model generation started: false.
