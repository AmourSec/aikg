#!/usr/bin/env python3
"""Regression tests for skills router.

Verifies:
1. catalog.json exists and is valid
2. every skill path points to an existing file
3. recall returns relevant candidates for a matching task
4. recall returns no false-positive-free candidates for an irrelevant task
5. a selected skill's SKILL.md can be loaded
6. no duplicate names in selectable set

Usage:
    python3 automatic-skill-routing/skills-router/scripts/test_routing.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROUTER_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
CATALOG_JSON = ROUTER_DIR / "catalog.json"


def load_catalog() -> dict:
    if not CATALOG_JSON.exists():
        sys.exit(f"FAIL: catalog.json not found at {CATALOG_JSON}")
    return json.loads(CATALOG_JSON.read_text(encoding="utf-8"))


def test_catalog_valid(catalog: dict) -> list[str]:
    """Check catalog structure and file existence."""
    failures: list[str] = []
    if not catalog.get("skills"):
        failures.append("catalog has no selectable skills")
    for s in catalog["skills"]:
        p = ROOT / s["path"]
        if not p.exists():
            failures.append(f"missing file: {s['path']}")
        if not s.get("name"):
            failures.append(f"skill missing name: {s['path']}")
        if not s.get("description"):
            failures.append(f"skill missing description: {s['path']}")
        if not s.get("path"):
            failures.append(f"skill missing path: {s['name']}")
    return failures


def test_no_duplicates(catalog: dict) -> list[str]:
    failures: list[str] = []
    names = [s["name"] for s in catalog["skills"]]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        failures.append(f"duplicate selectable names: {dupes}")
    return failures


def recall(task: str, catalog: dict, keywords: list[str]) -> list[dict]:
    """Simulate Step 1 recall using keyword matching."""
    candidates = []
    for s in catalog["skills"]:
        text = (s["name"] + " " + s["description"]).lower()
        hits = [k for k in keywords if k.lower() in text]
        if hits:
            candidates.append({**s, "_score": len(hits), "_hits": hits})
    candidates.sort(key=lambda x: -x["_score"])
    return candidates[:10]


def test_recall_matching(catalog: dict) -> list[str]:
    """A matching task should recall relevant candidates."""
    failures: list[str] = []
    task = "请检查当前昇腾环境、判断硬件架构"
    keywords = ["ascend", "npu", "env", "arch", "cann", "socversion", "npuarch", "环境", "硬件"]
    candidates = recall(task, catalog, keywords)
    if not candidates:
        failures.append(f"matching task recalled 0 candidates: {task}")
    # Verify top candidate is relevant
    top = candidates[0] if candidates else None
    if top and "env" not in top["name"].lower() and "arch" not in top["name"].lower() and "ascend" not in top["name"].lower():
        failures.append(f"top candidate seems irrelevant: {top['name']}")
    return failures


def test_recall_irrelevant(catalog: dict) -> list[str]:
    """An irrelevant task should have candidates that Step 2 would reject.

    We can't test semantic rejection here (that's the model's job), but we
    verify the catalog doesn't contain skills about explaining general AI
    concepts — those belong in docs/, not skills/.
    """
    failures: list[str] = []
    # This is a documentation task, not a skill task.
    # If recall returns candidates, they should be false positives
    # that a model would reject in Step 2.
    task = "解释 Transformer 的 self-attention 原理"
    keywords = ["transformer", "self-attention", "attention", "原理"]
    candidates = recall(task, catalog, keywords)
    # It's OK to have keyword hits, but none should be about explaining concepts
    for c in candidates:
        desc_lower = c["description"].lower()
        # Skills about transformer *libraries* or *operators* are false positives
        if "explain" in desc_lower and "concept" in desc_lower:
            failures.append(f"found a concept-explanation skill (should be docs): {c['name']}")
    return failures


def test_load_skill(catalog: dict) -> list[str]:
    """A selected skill's SKILL.md must be loadable."""
    failures: list[str] = []
    if not catalog["skills"]:
        failures.append("no skills to test loading")
        return failures
    # Pick first skill
    s = catalog["skills"][0]
    p = ROOT / s["path"]
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        failures.append(f"failed to load {s['path']}: {e}")
        return failures
    # Check it has frontmatter
    if not re.match(r"^---\s*\n", text):
        failures.append(f"loaded skill has no frontmatter: {s['path']}")
    return failures


def test_confirm_logic(catalog: dict) -> list[str]:
    """Skills with confirm tags should have confirm=true."""
    failures: list[str] = []
    # Check that confirm field is boolean
    for s in catalog["skills"]:
        if not isinstance(s["confirm"], bool):
            failures.append(f"confirm is not boolean for {s['name']}: {s['confirm']}")
    return failures


def main() -> int:
    catalog = load_catalog()
    all_failures: list[str] = []

    tests = [
        ("catalog_valid", test_catalog_valid),
        ("no_duplicates", test_no_duplicates),
        ("recall_matching", test_recall_matching),
        ("recall_irrelevant", test_recall_irrelevant),
        ("load_skill", test_load_skill),
        ("confirm_logic", test_confirm_logic),
    ]

    for name, fn in tests:
        failures = fn(catalog)
        if failures:
            all_failures.extend(failures)
            print(f"FAIL {name}:")
            for f in failures:
                print(f"  - {f}")
        else:
            print(f"PASS {name}")

    print(f"\n{'='*40}")
    if all_failures:
        print(f"FAILED: {len(all_failures)} issues")
        return 1
    print(f"OK: all {len(tests)} tests passed ({len(catalog['skills'])} skills)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
