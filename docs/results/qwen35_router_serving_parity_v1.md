# Qwen3.5 Router Serving Parity v1 Result

## Verdict

**NAMESPACE ROOT CAUSE SUPPORTED.**

The three arms use the same 192 already-observed SFT validation rows and
unchanged generation contract. The remapped adapter changes only tensor key
names; all 224 tensor dtype/shape/content hashes are unchanged.

## Summaries

```json
{
  "base": {
    "accuracy": 0.5833333333333334,
    "by_family": {
      "router_a": {
        "exact": 24,
        "samples": 64
      },
      "router_b": {
        "exact": 59,
        "samples": 64
      },
      "router_c": {
        "exact": 29,
        "samples": 64
      }
    },
    "exact": 112,
    "samples": 192
  },
  "original": {
    "accuracy": 0.5833333333333334,
    "by_family": {
      "router_a": {
        "exact": 24,
        "samples": 64
      },
      "router_b": {
        "exact": 59,
        "samples": 64
      },
      "router_c": {
        "exact": 29,
        "samples": 64
      }
    },
    "exact": 112,
    "samples": 192
  },
  "remapped": {
    "accuracy": 1.0,
    "by_family": {
      "router_a": {
        "exact": 64,
        "samples": 64
      },
      "router_b": {
        "exact": 64,
        "samples": 64
      },
      "router_c": {
        "exact": 64,
        "samples": 64
      }
    },
    "exact": 192,
    "samples": 192
  }
}
```

## HF Output Matches

```json
{
  "base": 112,
  "original": 112,
  "remapped": 192
}
```

## Original vs Remapped

```json
{
  "both_exact": 112,
  "both_wrong": 0,
  "original_only": 0,
  "remapped_only": 80
}
```

## Frozen Gates

```json
{
  "all_outputs_parseable_192": true,
  "all_three_arms_complete_192": true,
  "original_hf_output_match_less_than_192": true,
  "remap_tensor_content_unchanged": true,
  "remapped_each_label_exact_64": true,
  "remapped_exact_192": true,
  "remapped_exact_greater_than_original": true,
  "remapped_hf_output_match_192": true
}
```

## Boundaries

Passing does not revive observed integration v1. It permits only a separately
pre-registered, new history-disjoint integration v2. Real question scan,
benchmark, canary, holdout, training, and RL remain closed.

## Evidence

- prereg SHA: `be4bece33f078c64f25a831de279bf5ba9e037e4c8871db55a367a4039c6f47c`;
- service SHA: `1f741b74dfacc8e1845c18e1b08e609725f56a921ec8a913e16bd35436114d61`;
- raw result SHA: `52eed79199466bd3001134476f566eec9e7249a4ebbd4924edb3af2de2a949f0`;
- original adapter SHA: `48d72666cf269f64825791801c71aad5ae1d9e97cbc70574c216b12974e46e63`;
- remapped adapter SHA: `fbaa39dcb3fcf34e9aab280308cb5a5416094c1968e4ac3a69cd739a806ecc49`.
