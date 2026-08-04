#!/usr/bin/env python3
"""Validate skills-router/catalog.json for integrity and consistency.

Checks:
- catalog.json exists and is valid JSON
- every skill path points to an existing file
- no duplicate names
- frontmatter still parses and matches catalog metadata
- sources.yaml and catalog.json sources list is consistent

Usage:
    python3 skills-router/scripts/validate_catalog.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")

ROUTER_DIR = Path(__file__).resolve().parents[1]        # skills-router/
ROOT = Path(__file__).resolve().parents[3]              # 仓库根
CATALOG_JSON = ROUTER_DIR / "catalog.json"
SOURCES_YAML = ROUTER_DIR / "config" / "sources.yaml"


def parse_frontmatter(text: str) -> dict | None:
    import re
    m = re.match(r"^---\s*\n(.*?\n)---\s*\n", text, re.DOTALL)
    if not m:
        return None
    try:
        return yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None


def main() -> int:
    if not CATALOG_JSON.exists():
        sys.exit(f"catalog.json not found: {CATALOG_JSON}. Run build_catalog.py first.")

    try:
        catalog = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"catalog.json is invalid JSON: {e}")

    problems: list[str] = []

    # Check 1: sources.yaml consistency
    if SOURCES_YAML.exists():
        cfg = yaml.safe_load(SOURCES_YAML.read_text(encoding="utf-8")) or {}
        yaml_sources = {s["name"] for s in cfg.get("sources", [])}
        cat_sources = {s["name"] for s in catalog.get("sources", [])}
        if yaml_sources != cat_sources:
            problems.append(
                f"sources mismatch: sources.yaml={yaml_sources} catalog={cat_sources}"
            )

    # Check 2: skill paths exist
    for skill in catalog.get("skills", []):
        path = ROOT / skill["path"]
        if not path.exists():
            problems.append(f"missing file: {skill['path']}")
            continue
        meta = parse_frontmatter(path.read_text(encoding="utf-8"))
        if meta is None:
            problems.append(f"frontmatter unparseable: {skill['path']}")
            continue
        if meta.get("name") != skill["name"]:
            problems.append(
                f"name drift: catalog='{skill['name']}' file='{meta.get('name')}' at {skill['path']}"
            )

    # Check 3: no duplicate names among selectable
    names = [s["name"] for s in catalog.get("skills", [])]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        problems.append(f"duplicate selectable names: {dupes}")

    # Check 4: conflicts and errors are informational (from remote repos)
    warnings: list[str] = []
    if catalog.get("conflicts"):
        warnings.append(
            f"{len(catalog['conflicts'])} conflicts (excluded from selection, see catalog.json)"
        )
    if catalog.get("errors"):
        warnings.append(
            f"{len(catalog['errors'])} errors (excluded from selection, see catalog.json)"
        )

    for w in warnings:
        print(f"  note: {w}")

    if problems:
        print("VALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"OK: {len(names)} skills, 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
