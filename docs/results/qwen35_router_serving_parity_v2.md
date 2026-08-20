# Qwen3.5 Router Serving Parity v2 Result

## Verdict

**EXACT VLLM/HF PARITY.**

The content-identical namespace-remapped adapter was evaluated on all 1,536
already-observed SFT validation rows. V1 already established the namespace
root cause, so this run does not repeat the inert original-namespace arm.

## Summary

```json
{
  "accuracy": 1.0,
  "by_family": {
    "router_a": {
      "exact": 512,
      "hf_output_matches": 512,
      "samples": 512
    },
    "router_b": {
      "exact": 512,
      "hf_output_matches": 512,
      "samples": 512
    },
    "router_c": {
      "exact": 512,
      "hf_output_matches": 512,
      "samples": 512
    }
  },
  "c_by_subtype": {
    "box_total": {
      "exact": 64,
      "hf_output_matches": 64,
      "samples": 64
    },
    "paired_average": {
      "exact": 64,
      "hf_output_matches": 64,
      "samples": 64
    },
    "percentage_change": {
      "exact": 64,
      "hf_output_matches": 64,
      "samples": 64
    },
    "quotient_remainder": {
      "exact": 64,
      "hf_output_matches": 64,
      "samples": 64
    },
    "remaining_stock": {
      "exact": 64,
      "hf_output_matches": 64,
      "samples": 64
    },
    "single_operation": {
      "exact": 64,
      "hf_output_matches": 64,
      "samples": 64
    },
    "time_conversion": {
      "exact": 64,
      "hf_output_matches": 64,
      "samples": 64
    },
    "weighted_total": {
      "exact": 64,
      "hf_output_matches": 64,
      "samples": 64
    }
  },
  "exact": 1536,
  "hf_output_matches": 1536,
  "samples": 1536
}
```

## Frozen Gates

```json
{
  "all_outputs_parseable_1536": true,
  "each_c_subtype_exact_and_hf_match_64": true,
  "each_label_exact_and_hf_match_512": true,
  "remap_tensor_content_unchanged": true,
  "remapped_complete_1536": true,
  "remapped_exact_1536": true,
  "remapped_hf_output_match_1536": true
}
```

## Boundaries

Passing permits only a separately pre-registered, history-disjoint integration
v3. Integration generation, real question scan, benchmark, canary, holdout,
training, and RL remain closed.

## Evidence

- prereg SHA: `0ab39b8deb0269b9db171660c47bd16fe6fcd5bb8883b3879eacbc826ecece63`;
- service SHA: `c7ba00646eba567646c35fe5d558bf60eb125f688791125d3aa92dde585b17be`;
- raw result SHA: `df856042081d61a17ec71cb90ef47e83ebe8460084c901f770eda66347c9bc94`;
- original adapter SHA: `c40b6ab8811f162b60e6e290c7a9defbc0c013b239282bfbe8313eb986524c04`;
- remapped adapter SHA: `cea357d281ed100437268e213564fc5a5c00e6024b0c7a4be207cc686453e3f9`.
