# Qwen3.5 Router Skill Registry v5 Result

## Verdict

**ADMIT.**

```json
{
  "four_b_direct": {
    "accuracy": 0.31875,
    "by_family": {
      "box_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "first_strict_profit_period": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "implicit_scale_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "paired_average": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "percentage_change": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "quotient_remainder": {
        "cases": 16,
        "correct": 9,
        "parseable": 16
      },
      "remaining_stock": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "single_operation": {
        "cases": 16,
        "correct": 10,
        "parseable": 16
      },
      "time_conversion": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "weighted_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      }
    },
    "cases": 160,
    "correct": 51,
    "parseable": 160
  },
  "four_b_skill_registry_v5": {
    "accuracy": 1.0,
    "by_family": {
      "box_total": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "first_strict_profit_period": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "implicit_scale_total": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "paired_average": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "percentage_change": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "quotient_remainder": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "remaining_stock": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "single_operation": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "time_conversion": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      },
      "weighted_total": {
        "cases": 16,
        "correct": 16,
        "parseable": 16
      }
    },
    "cases": 160,
    "correct": 160,
    "parseable": 160
  },
  "nine_b_direct": {
    "accuracy": 0.3,
    "by_family": {
      "box_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "first_strict_profit_period": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "implicit_scale_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "paired_average": {
        "cases": 16,
        "correct": 15,
        "parseable": 16
      },
      "percentage_change": {
        "cases": 16,
        "correct": 15,
        "parseable": 16
      },
      "quotient_remainder": {
        "cases": 16,
        "correct": 7,
        "parseable": 16
      },
      "remaining_stock": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      },
      "single_operation": {
        "cases": 16,
        "correct": 10,
        "parseable": 16
      },
      "time_conversion": {
        "cases": 16,
        "correct": 1,
        "parseable": 16
      },
      "weighted_total": {
        "cases": 16,
        "correct": 0,
        "parseable": 16
      }
    },
    "cases": 160,
    "correct": 48,
    "parseable": 160
  }
}
```

```json
{
  "candidate_vs_four_b": {
    "baseline_accuracy": 0.31875,
    "baseline_only_case_ids": [],
    "bootstrap_samples": 10000,
    "bootstrap_seed": "qwen35-router-skill-registry-v5:candidate-four",
    "candidate_accuracy": 1.0,
    "candidate_only_case_ids": [
      "router-skill-v5-box_total-176e012803e8e1de",
      "router-skill-v5-box_total-301c395c9bcdc2cb",
      "router-skill-v5-box_total-3db333c79f9a28ff",
      "router-skill-v5-box_total-3e8fb1b7175c66b9",
      "router-skill-v5-box_total-3ec1478ce19131fe",
      "router-skill-v5-box_total-3f07675058863dae",
      "router-skill-v5-box_total-50f2e94cde6c7854",
      "router-skill-v5-box_total-5a760d19ff81165d",
      "router-skill-v5-box_total-5fd33ece5ae3db81",
      "router-skill-v5-box_total-6dd7617173747832",
      "router-skill-v5-box_total-751cfe6dd5d409c9",
      "router-skill-v5-box_total-79da5404dea7ab91",
      "router-skill-v5-box_total-9f7bc22a5f4d20c8",
      "router-skill-v5-box_total-a3c8c6614e66b411",
      "router-skill-v5-box_total-cadc74f7952a0989",
      "router-skill-v5-box_total-fd2c7830b2db8bf0",
      "router-skill-v5-first_strict_profit_period-1d245b61aead1a76",
      "router-skill-v5-first_strict_profit_period-2f52014d424fe3c5",
      "router-skill-v5-first_strict_profit_period-3283befda17eee23",
      "router-skill-v5-first_strict_profit_period-49a3739a4e746564",
      "router-skill-v5-first_strict_profit_period-4fb745514e2a8b39",
      "router-skill-v5-first_strict_profit_period-602164e53c8ed592",
      "router-skill-v5-first_strict_profit_period-6302b8c5e6a5102c",
      "router-skill-v5-first_strict_profit_period-7fd6391588a8963d",
      "router-skill-v5-first_strict_profit_period-8b65351d7559089b",
      "router-skill-v5-first_strict_profit_period-8c91e8afbfc69953",
      "router-skill-v5-first_strict_profit_period-a74e140338978ef8",
      "router-skill-v5-first_strict_profit_period-b5d121101faacca2",
      "router-skill-v5-first_strict_profit_period-c2048053fc0fe567",
      "router-skill-v5-first_strict_profit_period-c57e84d11dafeb2a",
      "router-skill-v5-first_strict_profit_period-e73e4bc3ac080d09",
      "router-skill-v5-first_strict_profit_period-f2b2ef5b59b4bc7b",
      "router-skill-v5-implicit_scale_total-05adef86309808e0",
      "router-skill-v5-implicit_scale_total-0ea367d5ee9ef009",
      "router-skill-v5-implicit_scale_total-1892a178b4c7bf32",
      "router-skill-v5-implicit_scale_total-526b051575c5c75d",
      "router-skill-v5-implicit_scale_total-5958eae429c92914",
      "router-skill-v5-implicit_scale_total-624baa71c9e35ca2",
      "router-skill-v5-implicit_scale_total-7841367a4e65389a",
      "router-skill-v5-implicit_scale_total-788ae2b9b93c8474",
      "router-skill-v5-implicit_scale_total-7b6cb08d8bbdc709",
      "router-skill-v5-implicit_scale_total-acde2c2ce207ab1e",
      "router-skill-v5-implicit_scale_total-b889944cb120c089",
      "router-skill-v5-implicit_scale_total-baf2ae3d14140e4d",
      "router-skill-v5-implicit_scale_total-c54ef4b8abe476e2",
      "router-skill-v5-implicit_scale_total-dc6c6cefaa9a1950",
      "router-skill-v5-implicit_scale_total-e23d42c5bdce3291",
      "router-skill-v5-implicit_scale_total-ea70bfcd0bcc8115",
      "router-skill-v5-quotient_remainder-1fe967f610f92b9b",
      "router-skill-v5-quotient_remainder-6216f2bf3f83b3aa",
      "router-skill-v5-quotient_remainder-776e2d74ea788fd0",
      "router-skill-v5-quotient_remainder-7a7be35de22875da",
      "router-skill-v5-quotient_remainder-c3fb23fa99443f05",
      "router-skill-v5-quotient_remainder-cc65b73bdb544064",
      "router-skill-v5-quotient_remainder-dfae432cb73ba8db",
      "router-skill-v5-remaining_stock-1898fd9944957ad1",
      "router-skill-v5-remaining_stock-1bca831b094a8956",
      "router-skill-v5-remaining_stock-3a6d1954dff920f4",
      "router-skill-v5-remaining_stock-3d3ebe0f6675a21b",
      "router-skill-v5-remaining_stock-55424b4b4f236a57",
      "router-skill-v5-remaining_stock-5d19a62452e0b5ea",
      "router-skill-v5-remaining_stock-603759cb8e2a2b18",
      "router-skill-v5-remaining_stock-86ed47d715cb3fca",
      "router-skill-v5-remaining_stock-ae6064b3c35eafe3",
      "router-skill-v5-remaining_stock-bccb8c386c730da8",
      "router-skill-v5-remaining_stock-c8be8310896eb34d",
      "router-skill-v5-remaining_stock-dec3020ec4508f3a",
      "router-skill-v5-remaining_stock-e251f4ca06b27774",
      "router-skill-v5-remaining_stock-e8e99e49ebfb4498",
      "router-skill-v5-remaining_stock-f60774359015d260",
      "router-skill-v5-remaining_stock-fdec48c19f8df432",
      "router-skill-v5-single_operation-50925e1954aafc55",
      "router-skill-v5-single_operation-88037e9d55f7348f",
      "router-skill-v5-single_operation-8acbc632e54de8b5",
      "router-skill-v5-single_operation-a7623db4e97bd756",
      "router-skill-v5-single_operation-b4b87313448289cf",
      "router-skill-v5-single_operation-e0528cea607a10f0",
      "router-skill-v5-time_conversion-20147480e4ba3d39",
      "router-skill-v5-time_conversion-31accfb766a96bbf",
      "router-skill-v5-time_conversion-4a32015511317cd7",
      "router-skill-v5-time_conversion-508614d02b4294dc",
      "router-skill-v5-time_conversion-5ef94cb59ed7963c",
      "router-skill-v5-time_conversion-5f0e228351c1cf36",
      "router-skill-v5-time_conversion-64a88a0b63263938",
      "router-skill-v5-time_conversion-6560e582348c6dda",
      "router-skill-v5-time_conversion-6d1511286509d0e2",
      "router-skill-v5-time_conversion-73d58d51c5d2c277",
      "router-skill-v5-time_conversion-742f2986196c4a56",
      "router-skill-v5-time_conversion-7f78baa8e979ca4b",
      "router-skill-v5-time_conversion-80341de394ae8102",
      "router-skill-v5-time_conversion-a0af688c9478f9b5",
      "router-skill-v5-time_conversion-bff78156d90784ed",
      "router-skill-v5-time_conversion-dcb149c0b94edb76",
      "router-skill-v5-weighted_total-05618ec7ac2907a1",
      "router-skill-v5-weighted_total-10f9f28a5d528b3d",
      "router-skill-v5-weighted_total-154f1b695038e030",
      "router-skill-v5-weighted_total-2acba7ef9cdbe7d3",
      "router-skill-v5-weighted_total-38e4a80135935944",
      "router-skill-v5-weighted_total-3e7aa3662e22f767",
      "router-skill-v5-weighted_total-6a8cbf3d4fc72628",
      "router-skill-v5-weighted_total-8d1849e03e867a11",
      "router-skill-v5-weighted_total-8fc4a1bda80af82d",
      "router-skill-v5-weighted_total-95c4728ad03445ab",
      "router-skill-v5-weighted_total-a3df3549590c98a6",
      "router-skill-v5-weighted_total-ba647a64170d7770",
      "router-skill-v5-weighted_total-c2c59cd473a20e73",
      "router-skill-v5-weighted_total-d71932164717c3a6",
      "router-skill-v5-weighted_total-f7334c2a94621a4c",
      "router-skill-v5-weighted_total-f7e238c1af318972"
    ],
    "cases": 160,
    "delta": 0.68125,
    "mcnemar_exact_p": 3.0814879110195774e-33,
    "paired_bootstrap_95_ci": [
      0.60625,
      0.75
    ],
    "paired_counts": {
      "baseline_only": 0,
      "both_correct": 51,
      "both_wrong": 0,
      "candidate_only": 109
    }
  },
  "candidate_vs_nine_b": {
    "baseline_accuracy": 0.3,
    "baseline_only_case_ids": [],
    "bootstrap_samples": 10000,
    "bootstrap_seed": "qwen35-router-skill-registry-v5:candidate-nine",
    "candidate_accuracy": 1.0,
    "candidate_only_case_ids": [
      "router-skill-v5-box_total-176e012803e8e1de",
      "router-skill-v5-box_total-301c395c9bcdc2cb",
      "router-skill-v5-box_total-3db333c79f9a28ff",
      "router-skill-v5-box_total-3e8fb1b7175c66b9",
      "router-skill-v5-box_total-3ec1478ce19131fe",
      "router-skill-v5-box_total-3f07675058863dae",
      "router-skill-v5-box_total-50f2e94cde6c7854",
      "router-skill-v5-box_total-5a760d19ff81165d",
      "router-skill-v5-box_total-5fd33ece5ae3db81",
      "router-skill-v5-box_total-6dd7617173747832",
      "router-skill-v5-box_total-751cfe6dd5d409c9",
      "router-skill-v5-box_total-79da5404dea7ab91",
      "router-skill-v5-box_total-9f7bc22a5f4d20c8",
      "router-skill-v5-box_total-a3c8c6614e66b411",
      "router-skill-v5-box_total-cadc74f7952a0989",
      "router-skill-v5-box_total-fd2c7830b2db8bf0",
      "router-skill-v5-first_strict_profit_period-1d245b61aead1a76",
      "router-skill-v5-first_strict_profit_period-2f52014d424fe3c5",
      "router-skill-v5-first_strict_profit_period-3283befda17eee23",
      "router-skill-v5-first_strict_profit_period-49a3739a4e746564",
      "router-skill-v5-first_strict_profit_period-4fb745514e2a8b39",
      "router-skill-v5-first_strict_profit_period-602164e53c8ed592",
      "router-skill-v5-first_strict_profit_period-6302b8c5e6a5102c",
      "router-skill-v5-first_strict_profit_period-7fd6391588a8963d",
      "router-skill-v5-first_strict_profit_period-8b65351d7559089b",
      "router-skill-v5-first_strict_profit_period-8c91e8afbfc69953",
      "router-skill-v5-first_strict_profit_period-a74e140338978ef8",
      "router-skill-v5-first_strict_profit_period-b5d121101faacca2",
      "router-skill-v5-first_strict_profit_period-c2048053fc0fe567",
      "router-skill-v5-first_strict_profit_period-c57e84d11dafeb2a",
      "router-skill-v5-first_strict_profit_period-e73e4bc3ac080d09",
      "router-skill-v5-first_strict_profit_period-f2b2ef5b59b4bc7b",
      "router-skill-v5-implicit_scale_total-05adef86309808e0",
      "router-skill-v5-implicit_scale_total-0ea367d5ee9ef009",
      "router-skill-v5-implicit_scale_total-1892a178b4c7bf32",
      "router-skill-v5-implicit_scale_total-526b051575c5c75d",
      "router-skill-v5-implicit_scale_total-5958eae429c92914",
      "router-skill-v5-implicit_scale_total-624baa71c9e35ca2",
      "router-skill-v5-implicit_scale_total-7841367a4e65389a",
      "router-skill-v5-implicit_scale_total-788ae2b9b93c8474",
      "router-skill-v5-implicit_scale_total-7b6cb08d8bbdc709",
      "router-skill-v5-implicit_scale_total-acde2c2ce207ab1e",
      "router-skill-v5-implicit_scale_total-b889944cb120c089",
      "router-skill-v5-implicit_scale_total-baf2ae3d14140e4d",
      "router-skill-v5-implicit_scale_total-c54ef4b8abe476e2",
      "router-skill-v5-implicit_scale_total-dc6c6cefaa9a1950",
      "router-skill-v5-implicit_scale_total-e23d42c5bdce3291",
      "router-skill-v5-implicit_scale_total-ea70bfcd0bcc8115",
      "router-skill-v5-paired_average-cbb62fb5d275b290",
      "router-skill-v5-percentage_change-b204c76d2fad3e5a",
      "router-skill-v5-quotient_remainder-128547ccb948066c",
      "router-skill-v5-quotient_remainder-1d33567bc2a380a5",
      "router-skill-v5-quotient_remainder-1fe967f610f92b9b",
      "router-skill-v5-quotient_remainder-408e9c682e3a44ee",
      "router-skill-v5-quotient_remainder-7a7be35de22875da",
      "router-skill-v5-quotient_remainder-98063b5509a2b694",
      "router-skill-v5-quotient_remainder-9f688b297eade01f",
      "router-skill-v5-quotient_remainder-cc65b73bdb544064",
      "router-skill-v5-quotient_remainder-eb65d816a33aef70",
      "router-skill-v5-remaining_stock-1898fd9944957ad1",
      "router-skill-v5-remaining_stock-1bca831b094a8956",
      "router-skill-v5-remaining_stock-3a6d1954dff920f4",
      "router-skill-v5-remaining_stock-3d3ebe0f6675a21b",
      "router-skill-v5-remaining_stock-55424b4b4f236a57",
      "router-skill-v5-remaining_stock-5d19a62452e0b5ea",
      "router-skill-v5-remaining_stock-603759cb8e2a2b18",
      "router-skill-v5-remaining_stock-86ed47d715cb3fca",
      "router-skill-v5-remaining_stock-ae6064b3c35eafe3",
      "router-skill-v5-remaining_stock-bccb8c386c730da8",
      "router-skill-v5-remaining_stock-c8be8310896eb34d",
      "router-skill-v5-remaining_stock-dec3020ec4508f3a",
      "router-skill-v5-remaining_stock-e251f4ca06b27774",
      "router-skill-v5-remaining_stock-e8e99e49ebfb4498",
      "router-skill-v5-remaining_stock-f60774359015d260",
      "router-skill-v5-remaining_stock-fdec48c19f8df432",
      "router-skill-v5-single_operation-3d79f381caadb722",
      "router-skill-v5-single_operation-88037e9d55f7348f",
      "router-skill-v5-single_operation-8acbc632e54de8b5",
      "router-skill-v5-single_operation-a7623db4e97bd756",
      "router-skill-v5-single_operation-b4b87313448289cf",
      "router-skill-v5-single_operation-e0528cea607a10f0",
      "router-skill-v5-time_conversion-20147480e4ba3d39",
      "router-skill-v5-time_conversion-31accfb766a96bbf",
      "router-skill-v5-time_conversion-4a32015511317cd7",
      "router-skill-v5-time_conversion-508614d02b4294dc",
      "router-skill-v5-time_conversion-5ef94cb59ed7963c",
      "router-skill-v5-time_conversion-5f0e228351c1cf36",
      "router-skill-v5-time_conversion-64a88a0b63263938",
      "router-skill-v5-time_conversion-6560e582348c6dda",
      "router-skill-v5-time_conversion-6d1511286509d0e2",
      "router-skill-v5-time_conversion-73d58d51c5d2c277",
      "router-skill-v5-time_conversion-742f2986196c4a56",
      "router-skill-v5-time_conversion-7f78baa8e979ca4b",
      "router-skill-v5-time_conversion-a0af688c9478f9b5",
      "router-skill-v5-time_conversion-bff78156d90784ed",
      "router-skill-v5-time_conversion-dcb149c0b94edb76",
      "router-skill-v5-weighted_total-05618ec7ac2907a1",
      "router-skill-v5-weighted_total-10f9f28a5d528b3d",
      "router-skill-v5-weighted_total-154f1b695038e030",
      "router-skill-v5-weighted_total-2acba7ef9cdbe7d3",
      "router-skill-v5-weighted_total-38e4a80135935944",
      "router-skill-v5-weighted_total-3e7aa3662e22f767",
      "router-skill-v5-weighted_total-6a8cbf3d4fc72628",
      "router-skill-v5-weighted_total-8d1849e03e867a11",
      "router-skill-v5-weighted_total-8fc4a1bda80af82d",
      "router-skill-v5-weighted_total-95c4728ad03445ab",
      "router-skill-v5-weighted_total-a3df3549590c98a6",
      "router-skill-v5-weighted_total-ba647a64170d7770",
      "router-skill-v5-weighted_total-c2c59cd473a20e73",
      "router-skill-v5-weighted_total-d71932164717c3a6",
      "router-skill-v5-weighted_total-f7334c2a94621a4c",
      "router-skill-v5-weighted_total-f7e238c1af318972"
    ],
    "cases": 160,
    "delta": 0.7,
    "mcnemar_exact_p": 3.851859888774472e-34,
    "paired_bootstrap_95_ci": [
      0.63125,
      0.76875
    ],
    "paired_counts": {
      "baseline_only": 0,
      "both_correct": 48,
      "both_wrong": 0,
      "candidate_only": 112
    }
  },
  "four_b_vs_nine_b": {
    "baseline_accuracy": 0.3,
    "baseline_only_case_ids": [
      "router-skill-v5-quotient_remainder-6216f2bf3f83b3aa",
      "router-skill-v5-quotient_remainder-776e2d74ea788fd0",
      "router-skill-v5-quotient_remainder-c3fb23fa99443f05",
      "router-skill-v5-quotient_remainder-dfae432cb73ba8db",
      "router-skill-v5-single_operation-50925e1954aafc55",
      "router-skill-v5-time_conversion-80341de394ae8102"
    ],
    "bootstrap_samples": 10000,
    "bootstrap_seed": "qwen35-router-skill-registry-v5:four-nine",
    "candidate_accuracy": 0.31875,
    "candidate_only_case_ids": [
      "router-skill-v5-paired_average-cbb62fb5d275b290",
      "router-skill-v5-percentage_change-b204c76d2fad3e5a",
      "router-skill-v5-quotient_remainder-128547ccb948066c",
      "router-skill-v5-quotient_remainder-1d33567bc2a380a5",
      "router-skill-v5-quotient_remainder-408e9c682e3a44ee",
      "router-skill-v5-quotient_remainder-98063b5509a2b694",
      "router-skill-v5-quotient_remainder-9f688b297eade01f",
      "router-skill-v5-quotient_remainder-eb65d816a33aef70",
      "router-skill-v5-single_operation-3d79f381caadb722"
    ],
    "cases": 160,
    "delta": 0.01875,
    "mcnemar_exact_p": 0.60723876953125,
    "paired_bootstrap_95_ci": [
      -0.03125,
      0.0625
    ],
    "paired_counts": {
      "baseline_only": 6,
      "both_correct": 42,
      "both_wrong": 103,
      "candidate_only": 9
    }
  }
}
```

```json
{
  "ab_verified_32": true,
  "all_three_arms_complete_and_parseable_160": true,
  "c_single_skill_verified_128": true,
  "candidate_vs_four_significant_zero_loss": true,
  "candidate_vs_nine_significant_zero_loss": true,
  "every_family_non_regression": true,
  "fallbacks_zero": true,
  "registry_unique_128": true,
  "router_correct_160": true
}
```

V1-V5 cannot be rerun. Benchmark generation remains closed until a separate
treatment transfer is pre-registered.
