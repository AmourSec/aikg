#!/usr/bin/env python3
"""Generate a slim, injectable router context from catalog.json.

Produces router-context.md containing:
1. A condensed routing protocol (Step 1-5 rules)
2. A compact skills catalog (name + description + path + confirm only)

This is the single artifact to inject into any LLM system prompt to enable
automatic skill routing. The model reads it and follows the protocol.

Drops triggers/tags from catalog.json to minimize context size.
The full catalog.json remains for auditing and tooling.

Usage:
    python3 automatic-skill-routing/skills-router/scripts/generate_router_context.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROUTER_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
CATALOG_JSON = ROUTER_DIR / "catalog.json"
OUTPUT = ROUTER_DIR / "router-context.md"

PROTOCOL_HEADER = """\
# Skills Router Context

This file is auto-generated. Do not edit manually.
Regenerate with: python3 automatic-skill-routing/skills-router/scripts/generate_router_context.py

---

## Routing Protocol

When a user describes a task, follow these steps:

### Step 1 — RECALL
Scan the Skills Catalog below. Identify candidates whose description
relates to the user's task. Use keyword matching AND semantic judgment.
Return at most 10 candidates. Prefer over-recalling; Step 2 will filter.

### Step 2 — SELECT
For each candidate, decide if it truly fits the task. You MAY return
0, 1, or N skills. For multiple skills, assign `order` (1, 2, 3...) to
indicate usage sequence. Give a `reason` for each selected and rejected
skill. Output this JSON (do not skip):

```json
{
  "selected": [
    {"name": "<skill-name>", "order": 1, "reason": "<why>"},
    {"name": "<skill-name>", "order": 2, "reason": "<why>"}
  ],
  "rejected": [
    {"name": "<skill-name>", "reason": "<why not>"}
  ],
  "confirm_required": false,
  "confirm_reason": ""
}
```

Rules:
- `selected[].name` MUST exist in the catalog below.
- If nothing fits, return empty `selected` and say so in natural language.
- `confirm_required` = true if ANY selected skill has `confirm: true`.
- Do NOT select skills solely on keyword hits. Use semantic judgment.

### Step 3 — NOTIFY / CONFIRM
Before loading any skill, output this message:

```
准备使用以下 Skills：
- <name>：<one-line purpose>（需确认：<reason>）   ← only if confirm: true
- <name>：<one-line purpose>
```

- If `confirm_required` is false: show the message, then continue to Step 4.
- If `confirm_required` is true: show the message, then STOP and wait for
  the user to explicitly agree (e.g., "继续" / "yes"). Do NOT proceed to
  Step 4 until the user agrees.
- If the user refuses a confirmed skill, remove it from selected and
  re-evaluate whether the remaining skills can complete the task.

### Step 4 — LOAD
For each selected skill (in order), read the file at its `path` to load
the full SKILL.md content. If the skill references `references/`,
`scripts/`, or `assets/` directories, read those on demand.

Only load selected skills. Do NOT load the entire catalog's full text.

### Step 5 — EXECUTE
Follow the loaded skill's own instructions to complete the task.
If a skill requires sub-step confirmations, follow its own rules.

---

## Skills Catalog

"""


def main() -> int:
    if not CATALOG_JSON.exists():
        sys.exit(f"catalog.json not found: {CATALOG_JSON}. Run build_catalog.py first.")

    catalog = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append(PROTOCOL_HEADER)
    lines.append(f"_Total: {len(catalog['skills'])} skills from "
                 f"{len([s for s in catalog['sources'] if s.get('enabled', True)])} "
                 f"active sources._")
    lines.append("")

    # Group by source for readability
    by_source: dict[str, list[dict]] = {}
    for skill in catalog["skills"]:
        by_source.setdefault(skill["source"], []).append(skill)

    for source_name in sorted(by_source.keys()):
        skills = by_source[source_name]
        lines.append(f"### Source: {source_name} ({len(skills)} skills)")
        lines.append("")
        for s in sorted(skills, key=lambda x: x["name"]):
            confirm_tag = " **[confirm]**" if s["confirm"] else ""
            lines.append(f"- **{s['name']}**{confirm_tag}")
            lines.append(f"  - path: `{s['path']}`")
            # Truncate very long descriptions to keep context manageable
            desc = s["description"]
            if len(desc) > 300:
                desc = desc[:297] + "..."
            lines.append(f"  - desc: {desc}")
            lines.append("")

    # Append conflicts and errors summary for transparency
    if catalog.get("conflicts"):
        lines.append("### Conflicts (excluded from selection)")
        lines.append("")
        for c in catalog["conflicts"]:
            lines.append(f"- `{c['name']}`: {len(c['paths'])} duplicate definitions")
        lines.append("")

    if catalog.get("errors"):
        lines.append("### Errors (excluded from selection)")
        lines.append("")
        for e in catalog["errors"]:
            lines.append(f"- {e['path']}: {e['reason']}")
        lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({size_kb:.0f} KB, "
          f"{len(catalog['skills'])} skills)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
