from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nano_harness.types import Task


REGISTRY_SCHEMA = "nano_harness_skill_registry_v1"
FRONTMATTER_PATTERN = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    name: str
    description: str
    instructions: str
    required_tags: frozenset[str]
    always: bool
    priority: int
    path: Path
    sha256: str

    def to_receipt(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "required_tags": sorted(self.required_tags),
            "always": self.always,
            "priority": self.priority,
            "sha256": self.sha256,
        }


class SkillRegistry:
    def __init__(
        self,
        *,
        registry_id: str,
        manifest_path: Path,
        skills: list[SkillDefinition],
    ):
        if not skills:
            raise ValueError("skill registry must contain at least one skill")
        self.registry_id = registry_id
        self.manifest_path = manifest_path
        self.skills = tuple(skills)
        self.by_id = {skill.skill_id: skill for skill in skills}
        if len(self.by_id) != len(skills):
            raise ValueError("skill registry IDs must be unique")
        self.sha256 = sha256_text(
            canonical_json(
                {
                    "registry_id": registry_id,
                    "skills": [skill.to_receipt() for skill in skills],
                }
            )
        )

    @classmethod
    def from_manifest(cls, path: str | Path) -> "SkillRegistry":
        manifest_path = Path(path).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != REGISTRY_SCHEMA:
            raise ValueError("unsupported skill registry schema")
        registry_id = str(manifest.get("registry_id", ""))
        if not registry_id:
            raise ValueError("skill registry needs a registry_id")
        entries = manifest.get("skills")
        if not isinstance(entries, list) or not entries:
            raise ValueError("skill registry needs skill entries")
        skills = [
            load_skill_definition(manifest_path.parent, entry)
            for entry in entries
        ]
        return cls(
            registry_id=registry_id,
            manifest_path=manifest_path,
            skills=skills,
        )

    def route(self, task: Task) -> tuple[list[SkillDefinition], dict[str, Any]]:
        raw_tags = task.metadata.get("skill_tags", [])
        if not isinstance(raw_tags, list) or any(
            not isinstance(tag, str) or not tag.strip() for tag in raw_tags
        ):
            raise ValueError("task metadata.skill_tags must be a string list")
        tags = frozenset(tag.strip().lower() for tag in raw_tags)
        selected = [
            skill
            for skill in self.skills
            if skill.always or skill.required_tags <= tags
        ]
        selected.sort(
            key=lambda skill: (
                0 if skill.always else 1,
                skill.priority,
                -len(skill.required_tags),
                skill.skill_id,
            )
        )
        receipt = {
            "schema_version": "nano_harness_skill_route_v1",
            "registry_id": self.registry_id,
            "registry_sha256": self.sha256,
            "task_id": task.task_id,
            "task_tags": sorted(tags),
            "selected_skills": [skill.to_receipt() for skill in selected],
        }
        return selected, receipt

    def render(self, selected: list[SkillDefinition]) -> str:
        if not selected:
            return ""
        sections = [
            (
                f"<skill id=\"{skill.skill_id}\" sha256=\"{skill.sha256}\">\n"
                f"{skill.instructions.strip()}\n"
                "</skill>"
            )
            for skill in selected
        ]
        return (
            "Apply the following routed skills in order. They are instructions, "
            "not evidence that an action succeeded.\n"
            f"Registry: {self.registry_id} sha256={self.sha256}\n\n"
            + "\n\n".join(sections)
        )


def load_skill_definition(
    manifest_root: Path,
    entry: dict[str, Any],
) -> SkillDefinition:
    if not isinstance(entry, dict):
        raise ValueError("skill registry entry must be an object")
    allowed = {
        "always",
        "path",
        "priority",
        "required_tags",
        "source_name",
        "skill_id",
    }
    unknown = set(entry) - allowed
    if unknown:
        raise ValueError(f"unknown skill registry fields: {sorted(unknown)}")
    skill_id = str(entry.get("skill_id", ""))
    if not re.fullmatch(r"[a-z0-9-]+", skill_id):
        raise ValueError("skill_id must be lowercase hyphen-case")
    relative = Path(str(entry.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("skill paths must stay under the manifest directory")
    skill_path = (manifest_root / relative).resolve()
    if manifest_root.resolve() not in skill_path.parents:
        raise ValueError("skill path escapes the manifest directory")
    content = skill_path.read_text(encoding="utf-8")
    frontmatter_match = FRONTMATTER_PATTERN.match(content)
    if frontmatter_match is None:
        raise ValueError(f"{skill_path} lacks YAML frontmatter")
    frontmatter = yaml.safe_load(frontmatter_match.group(1))
    if not isinstance(frontmatter, dict):
        raise ValueError("skill frontmatter must be an object")
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    source_name = str(entry.get("source_name", skill_id))
    if name != source_name:
        raise ValueError(
            f"registry source_name {source_name} does not match frontmatter {name}"
        )
    if not isinstance(description, str) or not description.strip():
        raise ValueError("skill description must not be empty")
    required_tags = entry.get("required_tags", [])
    if not isinstance(required_tags, list) or any(
        not isinstance(tag, str) or not tag.strip() for tag in required_tags
    ):
        raise ValueError("required_tags must be a string list")
    always = entry.get("always", False)
    if not isinstance(always, bool):
        raise ValueError("always must be boolean")
    priority = entry.get("priority", 100)
    if type(priority) is not int or priority < 0:
        raise ValueError("priority must be a non-negative integer")
    return SkillDefinition(
        skill_id=skill_id,
        name=name,
        description=description.strip(),
        instructions=content[frontmatter_match.end() :].strip(),
        required_tags=frozenset(
            tag.strip().lower() for tag in required_tags
        ),
        always=always,
        priority=priority,
        path=skill_path,
        sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
