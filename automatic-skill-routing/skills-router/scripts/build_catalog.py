#!/usr/bin/env python3
"""Build a lightweight skills catalog from configured sources.

Reads skills-router/config/sources.yaml, scans each source for SKILL.md files,
parses their frontmatter, and writes skills-router/catalog.json.

This script is a generic scanner. It does not hardcode any skill name.
Adding a new skill only requires dropping it into a source directory.
Adding a new source only requires appending to sources.yaml.

Usage:
    python3 skills-router/scripts/build_catalog.py
    python3 skills-router/scripts/build_catalog.py --check   # validate only, do not write
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")


ROUTER_DIR = Path(__file__).resolve().parents[1]        # skills-router/
ROOT = Path(__file__).resolve().parents[3]              # 仓库根
SOURCES_YAML = ROUTER_DIR / "config" / "sources.yaml"
CATALOG_JSON = ROUTER_DIR / "catalog.json"

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "for", "to", "of", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "as", "by", "with", "from", "when", "asked", "use",
    "used", "not", "no", "do", "does", "did", "will", "would", "should", "can",
    "could", "may", "might", "must", "shall", "you", "your", "we", "our", "they",
    "them", "their", "he", "she", "his", "her", "which", "what", "who", "whom",
    "how", "why", "where", "there", "here", "than", "then", "so", "if", "else",
    "while", "about", "into", "over", "under", "out", "up", "down", "all", "any",
    "some", "such", "more", "most", "other", "only", "own", "same", "very", "s",
    "t", "d", "ll", "m", "re", "ve", "y",
}


@dataclass
class CatalogError:
    source: str
    path: str
    reason: str


@dataclass
class CatalogConflict:
    name: str
    paths: list[str]


@dataclass
class CatalogSkill:
    name: str
    description: str
    source: str
    path: str
    tags: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    confirm: bool = False


def load_sources_config() -> dict[str, Any]:
    if not SOURCES_YAML.exists():
        sys.exit(f"sources.yaml not found: {SOURCES_YAML}")
    with SOURCES_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "sources" not in cfg:
        sys.exit("sources.yaml missing 'sources' key")
    return cfg


def sync_git_source(source: dict[str, Any]) -> tuple[bool, str]:
    """Clone or pull a git source into sync_dir. Returns (ok, message)."""
    sync_dir = ROOT / source["sync_dir"]
    url = source["url"]
    branch = source.get("branch", "main")
    if not sync_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "-b", branch, url, str(sync_dir)],
            check=False,
            capture_output=True,
        )
        if not sync_dir.exists():
            return False, f"git clone failed for {url}"
    else:
        subprocess.run(
            ["git", "-C", str(sync_dir), "pull", "--ff-only"],
            check=False,
            capture_output=True,
        )
    return True, datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_frontmatter(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse YAML frontmatter from SKILL.md. Returns (meta, error)."""
    m = re.match(r"^---\s*\n(.*?\n)---\s*\n", text, re.DOTALL)
    if not m:
        return None, "missing frontmatter delimiters"
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        return None, f"yaml parse error: {e}"
    return meta, None


def extract_triggers(description: str) -> list[str]:
    """Extract lowercase tokens from description as fallback triggers."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", description.lower())
    seen: list[str] = []
    for t in tokens:
        if t in STOP_WORDS or len(t) < 3:
            continue
        if t not in seen:
            seen.append(t)
    return seen[:12]


def scan_source(
    source: dict[str, Any],
    defaults: dict[str, Any],
) -> tuple[list[CatalogSkill], list[CatalogError], dict[str, Any]]:
    skills: list[CatalogSkill] = []
    errors: list[CatalogError] = []
    source_meta: dict[str, Any] = {"name": source["name"], "type": source["type"]}

    if source["type"] == "git":
        ok, msg = sync_git_source(source)
        if not ok:
            errors.append(CatalogError(source["name"], source["url"], msg))
            return skills, errors, source_meta
        source_meta["synced_at"] = msg
        scan_root = ROOT / source["sync_dir"]
    else:
        scan_root = ROOT / source["root"]

    if not scan_root.exists():
        errors.append(CatalogError(source["name"], str(scan_root), "root directory missing"))
        return skills, errors, source_meta

    if source["type"] == "git":
        source_meta["root"] = source["root"]
    else:
        source_meta["root"] = source["root"]

    min_chars = defaults.get("description_min_chars", 20)
    confirm_tags = set(defaults.get("confirm_tags", []))
    source_default_confirm = source.get("default_confirm", False)

    for skill_md in sorted(scan_root.rglob("SKILL.md")):
        rel_path = skill_md.relative_to(ROOT).as_posix()
        text = skill_md.read_text(encoding="utf-8")
        meta, err = parse_frontmatter(text)
        if err:
            errors.append(CatalogError(source["name"], rel_path, err))
            continue

        name = meta.get("name")
        description = meta.get("description")
        if not name:
            errors.append(CatalogError(source["name"], rel_path, "missing 'name'"))
            continue
        if not description:
            errors.append(CatalogError(source["name"], rel_path, "missing 'description'"))
            continue
        if not isinstance(description, str) or len(description) < min_chars:
            errors.append(CatalogError(
                source["name"], rel_path,
                f"description too short (<{min_chars} chars)",
            ))
            continue

        tags = meta.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        triggers = meta.get("triggers") or extract_triggers(description)

        confirm = bool(meta.get("confirm", source_default_confirm))
        if not confirm and any(t in confirm_tags for t in tags):
            confirm = True

        skills.append(CatalogSkill(
            name=name,
            description=description,
            source=source["name"],
            path=rel_path,
            tags=tags,
            triggers=triggers,
            confirm=confirm,
        ))

    return skills, errors, source_meta


def detect_conflicts(skills: list[CatalogSkill]) -> list[CatalogConflict]:
    by_name: dict[str, list[str]] = {}
    for s in skills:
        by_name.setdefault(s.name, []).append(s.path)
    return [
        CatalogConflict(name=name, paths=paths)
        for name, paths in by_name.items()
        if len(paths) > 1
    ]


def build_catalog() -> dict[str, Any]:
    cfg = load_sources_config()
    defaults = cfg.get("defaults", {})
    sources_cfg = cfg["sources"]

    all_skills: list[CatalogSkill] = []
    all_errors: list[CatalogError] = []
    sources_meta: list[dict[str, Any]] = []

    for source in sources_cfg:
        if source.get("enabled", True) is False:
            sources_meta.append({
                "name": source["name"],
                "type": source["type"],
                "enabled": False,
            })
            continue
        skills, errors, meta = scan_source(source, defaults)
        all_skills.extend(skills)
        all_errors.extend(errors)
        sources_meta.append(meta)

    conflicts = detect_conflicts(all_skills)

    # Remove conflicted skills from selectable set
    conflicted_names = {c.name for c in conflicts}
    selectable = [s for s in all_skills if s.name not in conflicted_names]

    return {
        "version": "1.0",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": sources_meta,
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "source": s.source,
                "path": s.path,
                "tags": s.tags,
                "triggers": s.triggers,
                "confirm": s.confirm,
            }
            for s in sorted(selectable, key=lambda x: (x.source, x.name))
        ],
        "conflicts": [
            {"name": c.name, "paths": c.paths} for c in conflicts
        ],
        "errors": [
            {"source": e.source, "path": e.path, "reason": e.reason}
            for e in all_errors
        ],
    }


def validate_catalog(catalog: dict[str, Any]) -> list[str]:
    """Return list of human-readable validation warnings."""
    warnings: list[str] = []
    if catalog["conflicts"]:
        for c in catalog["conflicts"]:
            warnings.append(
                f"conflict: skill '{c['name']}' defined in {len(c['paths'])} paths: {c['paths']}"
            )
    if catalog["errors"]:
        for e in catalog["errors"]:
            warnings.append(
                f"error in source '{e['source']}' at {e['path']}: {e['reason']}"
            )
    if not catalog["skills"]:
        warnings.append("warning: no selectable skills in catalog")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="validate only, do not write catalog.json")
    parser.add_argument("--print", action="store_true",
                        help="print catalog to stdout")
    args = parser.parse_args()

    catalog = build_catalog()
    warnings = validate_catalog(catalog)

    for w in warnings:
        print(w, file=sys.stderr)

    print(
        f"catalog: {len(catalog['skills'])} skills, "
        f"{len(catalog['conflicts'])} conflicts, "
        f"{len(catalog['errors'])} errors",
        file=sys.stderr,
    )

    if args.print:
        print(json.dumps(catalog, indent=2, ensure_ascii=False))
        return 0

    if not args.check:
        CATALOG_JSON.write_text(
            json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {CATALOG_JSON.relative_to(ROOT)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
