from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StateLedger:
    objective: str
    constraints: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def render(self, max_chars: int) -> str:
        sections = [
            ("OBJECTIVE", [self.objective]),
            ("CONSTRAINTS", self.constraints),
            ("FACTS", self.facts),
            ("COMPLETED", self.completed),
            ("PENDING", self.pending),
            ("FAILURES", self.failures),
            ("VERIFICATION EVIDENCE", self.evidence),
        ]
        lines: list[str] = ["<state_ledger>"]
        for title, items in sections:
            lines.append(f"{title}:")
            lines.extend(f"- {item}" for item in items[-12:])
        lines.append("</state_ledger>")
        rendered = "\n".join(lines)
        if len(rendered) <= max_chars:
            return rendered
        keep = max(0, max_chars - 80)
        return rendered[:keep] + "\n- [ledger truncated; latest evidence retained externally]\n"


def compact_messages(
    messages: list[dict],
    ledger: StateLedger,
    max_chars: int,
    reserve_chars: int,
    scratchpad_chars: int,
) -> list[dict]:
    budget = max(2000, max_chars - reserve_chars)
    ledger_message = {"role": "system", "content": ledger.render(scratchpad_chars)}
    first_assistant = next(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ),
        len(messages),
    )
    pinned = messages[:first_assistant]
    mutable_history = messages[first_assistant:]
    if sum(len(str(message.get("content", ""))) for message in messages) <= budget:
        return [*pinned, ledger_message, *mutable_history]
    tail: list[dict] = []
    used = sum(len(str(message.get("content", ""))) for message in pinned)
    used += len(ledger_message["content"])
    for message in reversed(mutable_history):
        size = len(str(message.get("content", ""))) + 200
        if used + size > budget:
            continue
        tail.append(message)
        used += size
    tail.reverse()
    summary = {
        "role": "system",
        "content": (
            "Earlier transcript content was compacted. The state ledger is the "
            "authoritative continuation state; do not assume omitted tool results."
        ),
    }
    return [*pinned, ledger_message, summary, *tail]
