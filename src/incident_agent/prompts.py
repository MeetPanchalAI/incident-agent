"""Loads the prompt text from the `prompts/` directory.

Prompts are files, not string literals, so they can be edited and re-measured
against the evaluation scenarios without touching the code. `AGENT_PROMPTS_DIR`
points somewhere else if you want to keep a variant alongside.

Placeholders use `$name` and are filled with `string.Template`, which leaves
anything it does not recognise untouched.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from .config import Settings, format_iso
from .tools.store import METRICS

FILES = ("system", "submit_not_alone", "submit_invalid", "force_final", "stopped_early")


def load(directory: Path, name: str) -> str:
    path = directory / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt '{name}' not found at {path}. Expected files: {', '.join(f'{n}.md' for n in FILES)}. "
            "Set AGENT_PROMPTS_DIR if your prompts live elsewhere."
        )
    return path.read_text(encoding="utf-8").strip()


class Prompts:
    """The prompt text for one agent, read once at construction."""

    def __init__(self, settings: Settings, services: list[str], dataset: str) -> None:
        self.directory = settings.prompts_dir
        self.system = Template(load(self.directory, "system")).safe_substitute(
            now=format_iso(settings.now),
            services=", ".join(services) or "none - no data has been ingested yet",
            metrics=", ".join(sorted(METRICS)),
            dataset=dataset,
        )
        self.submit_not_alone = load(self.directory, "submit_not_alone")
        self.force_final = load(self.directory, "force_final")
        self.stopped_early = load(self.directory, "stopped_early")
        self._submit_invalid = Template(load(self.directory, "submit_invalid"))

    def submit_invalid(self, errors: list[str]) -> str:
        return self._submit_invalid.safe_substitute(errors="\n".join(f"- {e}" for e in errors))
