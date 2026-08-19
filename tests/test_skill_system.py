from __future__ import annotations

import json
import tempfile
import unittest
import copy
from pathlib import Path

from nano_harness.client import ScriptedClient
from nano_harness.config import HarnessConfig, load_run_config
from nano_harness.harness import AgentHarness
from nano_harness.skill_evolution import (
    build_candidate_request,
    cluster_failures,
    evaluate_registry,
    load_contract_suite,
    select_candidate,
)
from nano_harness.skill_system import SkillRegistry
from nano_harness.types import ModelReply, Task


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "skills/registry_parent_v1.json"
CANDIDATE = ROOT / "skills/registry_candidate_v2.json"
SUITE = ROOT / "skills/synthetic_contract_suite_v1.json"


class SkillSystemTests(unittest.TestCase):
    def test_routes_core_and_minimal_tag_skill(self):
        registry = SkillRegistry.from_manifest(PARENT)
        task = Task(
            task_id="route-1",
            benchmark="synthetic",
            messages=[{"role": "user", "content": "Use a tool safely."}],
            metadata={"skill_tags": ["tool-use", "error-recovery"]},
        )

        selected, receipt = registry.route(task)

        self.assertEqual(
            [skill.skill_id for skill in selected],
            ["compact-agent-core", "tool-recovery"],
        )
        self.assertEqual(receipt["registry_sha256"], registry.sha256)
        self.assertEqual(receipt["task_tags"], ["error-recovery", "tool-use"])

    def test_rejects_unknown_manifest_field(self):
        raw = json.loads(PARENT.read_text(encoding="utf-8"))
        raw["skills"][0]["unknown"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown"):
                SkillRegistry.from_manifest(path)

    def test_candidate_improves_frozen_suite_without_regression(self):
        suite = load_contract_suite(SUITE)
        parent_registry = SkillRegistry.from_manifest(PARENT)
        candidate_registry = SkillRegistry.from_manifest(CANDIDATE)
        parent = evaluate_registry(parent_registry, suite)
        candidate = evaluate_registry(candidate_registry, suite)

        promotion = select_candidate(parent, candidate, suite)

        self.assertEqual((parent["passed"], parent["total"]), (4, 6))
        self.assertEqual((candidate["passed"], candidate["total"]), (6, 6))
        self.assertTrue(promotion["promoted"])
        self.assertAlmostEqual(promotion["aggregate_delta"], 2 / 6)
        self.assertEqual(promotion["reasons"], [])
        for family in suite["protected_families"]:
            self.assertGreaterEqual(
                candidate["by_family"][family]["score"],
                parent["by_family"][family]["score"],
            )

    def test_failure_clusters_drive_bounded_candidate_request(self):
        suite = load_contract_suite(SUITE)
        parent_registry = SkillRegistry.from_manifest(PARENT)
        parent = evaluate_registry(parent_registry, suite)

        request = build_candidate_request(parent_registry, parent, suite)
        clusters = cluster_failures(parent)

        self.assertEqual(
            {cluster["family"] for cluster in clusters},
            {"error-recovery"},
        )
        self.assertEqual(
            request["allowed_mutation_scope"],
            ["tool-recovery skill instructions"],
        )
        self.assertIn("benchmark prompts", request["forbidden_inputs"])
        self.assertEqual(
            request["case_ids_sha256"],
            parent["case_ids_sha256"],
        )
        self.assertEqual(
            request["acceptance"]["allowed_mutated_skill_ids"],
            ["tool-recovery"],
        )

    def test_rejects_candidate_with_different_cases(self):
        suite = load_contract_suite(SUITE)
        parent = evaluate_registry(SkillRegistry.from_manifest(PARENT), suite)
        candidate = evaluate_registry(
            SkillRegistry.from_manifest(CANDIDATE),
            suite,
        )
        candidate["case_ids_sha256"] = "0" * 64

        promotion = select_candidate(parent, candidate, suite)

        self.assertFalse(promotion["promoted"])
        self.assertIn("case_identity_mismatch", promotion["reasons"])

    def test_rejects_candidate_with_different_suite(self):
        suite = load_contract_suite(SUITE)
        parent = evaluate_registry(SkillRegistry.from_manifest(PARENT), suite)
        candidate = evaluate_registry(
            SkillRegistry.from_manifest(CANDIDATE),
            suite,
        )
        candidate["suite_sha256"] = "0" * 64

        promotion = select_candidate(parent, candidate, suite)

        self.assertFalse(promotion["promoted"])
        self.assertIn("suite_identity_mismatch", promotion["reasons"])

    def test_rejects_disallowed_core_skill_mutation(self):
        suite = load_contract_suite(SUITE)
        parent = evaluate_registry(SkillRegistry.from_manifest(PARENT), suite)
        candidate = evaluate_registry(
            SkillRegistry.from_manifest(CANDIDATE),
            suite,
        )
        candidate = copy.deepcopy(candidate)
        candidate["skill_sha256"]["compact-agent-core"] = "0" * 64

        promotion = select_candidate(parent, candidate, suite)

        self.assertFalse(promotion["promoted"])
        self.assertIn(
            "disallowed_skill_mutation:compact-agent-core",
            promotion["reasons"],
        )

    def test_rejects_protected_family_regression(self):
        suite = load_contract_suite(SUITE)
        parent = evaluate_registry(SkillRegistry.from_manifest(PARENT), suite)
        candidate = evaluate_registry(
            SkillRegistry.from_manifest(CANDIDATE),
            suite,
        )
        candidate = copy.deepcopy(candidate)
        candidate["by_family"]["core-safety"]["score"] = 0.5

        promotion = select_candidate(parent, candidate, suite)

        self.assertFalse(promotion["promoted"])
        self.assertIn(
            "protected_family_regression:core-safety",
            promotion["reasons"],
        )

    def test_skill_routed_harness_injects_prompt_and_receipt(self):
        registry = SkillRegistry.from_manifest(CANDIDATE)
        client = ScriptedClient([ModelReply(content="FINAL: done")])
        harness = AgentHarness(
            client,
            "test-model",
            HarnessConfig(strategy="skill_routed", audit_passes=0),
            skill_registry=registry,
        )
        task = Task(
            task_id="runtime-1",
            benchmark="synthetic",
            messages=[{"role": "user", "content": "Recover from a tool error."}],
            metadata={"skill_tags": ["tool-use", "error-recovery"]},
        )

        result = harness.run(task)

        self.assertEqual(result.status, "completed")
        route = result.trajectory[0]
        self.assertEqual(route["kind"], "skill_route")
        self.assertEqual(
            [row["skill_id"] for row in route["selected_skills"]],
            ["compact-agent-core", "tool-recovery"],
        )
        system_messages = [
            row["content"]
            for row in client.calls[0]["messages"]
            if row["role"] == "system"
        ]
        self.assertTrue(
            any("Classify the failure before retrying" in row for row in system_messages)
        )
        self.assertTrue(any(registry.sha256 in row for row in system_messages))

    def test_skill_routed_harness_requires_registry(self):
        harness = AgentHarness(
            ScriptedClient([ModelReply(content="unused")]),
            "test-model",
            HarnessConfig(strategy="skill_routed"),
        )
        task = Task(
            task_id="missing-registry",
            benchmark="synthetic",
            messages=[{"role": "user", "content": "test"}],
        )

        with self.assertRaisesRegex(ValueError, "requires"):
            harness.run(task)

    def test_run_config_resolves_skill_registry_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            registry_path = root / "registry.json"
            registry_path.write_text("{}", encoding="utf-8")
            config_path.write_text(
                "\n".join(
                    (
                        "model:",
                        "  name: test-model",
                        "harness:",
                        "  strategy: skill_routed",
                        "  skill_registry_path: registry.json",
                        "benchmark:",
                        "  name: synthetic",
                        "  source: inline",
                        "output_dir: results",
                        "run_id: skill-route",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_run_config(config_path)

            self.assertEqual(
                config.harness.skill_registry_path,
                str(registry_path.resolve()),
            )

    def test_base_harness_does_not_inject_skills(self):
        client = ScriptedClient([ModelReply(content="done")])
        harness = AgentHarness(
            client,
            "test-model",
            HarnessConfig(strategy="base"),
        )
        task = Task(
            task_id="base",
            benchmark="synthetic",
            messages=[{"role": "user", "content": "test"}],
            metadata={"skill_tags": ["tool-use"]},
        )

        result = harness.run(task)

        self.assertEqual(result.status, "completed")
        self.assertFalse(
            any(row.get("kind") == "skill_route" for row in result.trajectory)
        )
        self.assertFalse(
            any(
                "<skill " in row["content"]
                for row in client.calls[0]["messages"]
                if row["role"] == "system"
            )
        )


if __name__ == "__main__":
    unittest.main()
