from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, TypeAlias
from urllib.parse import urlsplit

import yaml

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
EXCLUDED_DIRECTORIES: Final = frozenset(
    {"99-templates", "superpowers", "13-contribution-leaderboard"}
)


@dataclass(frozen=True, slots=True)
class Article:
    source: str
    title: str
    authors: tuple[str, ...]
    views: int


@dataclass(frozen=True, slots=True)
class Snapshot:
    updated_at: str | None
    pages: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class LeaderboardDataError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def _clean_url(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme not in {"", "http", "https"} or not parsed.path.startswith("/"):
        return None
    if parsed.scheme and not parsed.netloc:
        return None
    if any(part in {".", ".."} for part in parsed.path.split("/")):
        return None
    if any(character.isspace() for character in value) or "\\" in value:
        return None
    path = parsed.path.rstrip("/") + "/"
    origin = (
        f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme else f"//{parsed.netloc}"
    )
    return f"{origin}{path}" if parsed.netloc else path


def load_snapshot(path: Path) -> Snapshot:
    payload: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LeaderboardDataError(f"{path}: snapshot must be an object")
    version = payload.get("schema_version")
    updated_at = payload.get("updated_at")
    pages = payload.get("pages")
    if type(version) is not int or version != 1:
        raise LeaderboardDataError(f"{path}: schema_version must be 1")
    if updated_at is not None and not isinstance(updated_at, str):
        raise LeaderboardDataError(f"{path}: updated_at must be a string or null")
    if updated_at is not None:
        try:
            timestamp = datetime.fromisoformat(updated_at)
        except ValueError as error:
            raise LeaderboardDataError(
                f"{path}: updated_at must be an ISO timestamp"
            ) from error
        if timestamp.utcoffset() is None:
            raise LeaderboardDataError(f"{path}: updated_at must include a timezone")
    if not isinstance(pages, dict):
        raise LeaderboardDataError(f"{path}: pages must be an object")
    counts: dict[str, int] = {}
    for source, count in pages.items():
        if type(count) is not int or count < 0:
            raise LeaderboardDataError(
                f"{path}: pageviews must be nonnegative integers"
            )
        normalized = _clean_url(source)
        if normalized is not None:
            counts[normalized] = counts.get(normalized, 0) + count
    return Snapshot(updated_at, MappingProxyType(counts))


def _nav_sources(nav: JsonValue) -> Iterator[str]:
    if isinstance(nav, str):
        yield nav
    if isinstance(nav, list):
        for entry in nav:
            yield from _nav_sources(entry)
    if isinstance(nav, dict):
        for entry in nav.values():
            yield from _nav_sources(entry)


def _article_source(value: str, root: Path) -> str | None:
    source = PurePosixPath(value)
    if source.is_absolute() or ".." in source.parts or source.suffix != ".md":
        return None
    if source.name == "index.md" or value == "knowledge-map.md":
        return None
    if value == "contribution-leaderboard.md":
        return None
    if EXCLUDED_DIRECTORIES.intersection(source.parts):
        return None
    docs = (root / "docs").resolve()
    target = (docs / source).resolve()
    if not target.is_relative_to(docs) or not target.is_file():
        return None
    return source.as_posix()


def _metadata(path: Path) -> tuple[Mapping[str, JsonValue], str]:
    content = path.read_text(encoding="utf-8-sig")
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content
    end = next(
        (
            index
            for index in range(1, len(lines))
            if lines[index].strip() in {"---", "..."}
        ),
        None,
    )
    if end is None:
        raise LeaderboardDataError(
            f"{path}: front matter is missing its closing delimiter"
        )
    try:
        metadata: JsonValue = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as error:
        raise LeaderboardDataError(f"{path}: invalid YAML front matter") from error
    if metadata is None:
        return {}, "\n".join(lines[end + 1 :])
    if not isinstance(metadata, dict):
        raise LeaderboardDataError(f"{path}: front matter must be a mapping")
    return metadata, "\n".join(lines[end + 1 :])


def _authors(metadata: Mapping[str, JsonValue], source: str) -> tuple[str, ...] | None:
    if "authors" in metadata:
        authors = metadata["authors"]
        if not isinstance(authors, list) or not authors:
            raise LeaderboardDataError(
                f"{source}: authors must be a nonempty list of names"
            )
        names: list[str] = []
        for author in authors:
            if not isinstance(author, str) or not author.strip():
                raise LeaderboardDataError(
                    f"{source}: authors must contain nonempty strings"
                )
            names.append(author.strip())
        return tuple(dict.fromkeys(names))
    if "author" in metadata:
        author = metadata["author"]
        if not isinstance(author, str) or not author.strip():
            raise LeaderboardDataError(f"{source}: author must be a nonempty string")
        return (author.strip(),)
    return None


def _git(root: Path, args: tuple[str, ...]) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise LeaderboardDataError(
            "Git history could not be read; run from a Git checkout with full history"
        )
    return result.stdout.strip()


def collect_articles(root: Path, snapshot: Snapshot) -> tuple[Article, ...]:
    if _git(root, ("rev-parse", "--is-shallow-repository")) == "true":
        raise LeaderboardDataError(
            "Article attribution requires complete Git history; use fetch-depth: 0 in checkout"
        )
    try:
        config: JsonValue = yaml.load(
            (root / "mkdocs.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
        )
    except yaml.YAMLError as error:
        raise LeaderboardDataError("mkdocs.yml: invalid YAML configuration") from error
    if not isinstance(config, dict):
        raise LeaderboardDataError("mkdocs.yml: configuration must be a mapping")
    site_url = config.get("site_url", "")
    if not isinstance(site_url, str):
        raise LeaderboardDataError("mkdocs.yml: site_url must be a string")
    site = urlsplit(site_url)
    base = site.path.rstrip("/")
    views: dict[str, int] = {}
    for url, count in snapshot.pages.items():
        normalized = _clean_url(url)
        if normalized is None:
            continue
        parsed = urlsplit(normalized)
        if parsed.netloc and parsed.netloc.lower() != site.netloc.lower():
            continue
        path = parsed.path
        if base and path.startswith(base + "/"):
            path = path[len(base) :]
        views[path] = views.get(path, 0) + count
    sources = {
        _article_source(value, root) for value in _nav_sources(config.get("nav"))
    }
    articles: list[Article] = []
    for source in sorted(value for value in sources if value is not None):
        metadata, body = _metadata(root / "docs" / source)
        authors = _authors(metadata, source)
        if authors is None:
            tracked = _git(root, ("ls-files", "--", f"docs/{source}"))
            creation = (
                _git(
                    root,
                    (
                        "log",
                        "--follow",
                        "--format=%aN",
                        "--diff-filter=A",
                        "--",
                        f"docs/{source}",
                    ),
                )
                if tracked
                else ""
            )
            names = tuple(
                name.strip() for name in creation.splitlines() if name.strip()
            )
            authors = names[-1:] if names else ()
        title = metadata.get("title")
        heading = re.search(r"^#\s+(.+?)\s*#*\s*$", body, re.MULTILINE)
        if not isinstance(title, str) or not title.strip():
            title = heading.group(1) if heading else PurePosixPath(source).stem
        articles.append(
            Article(source, title.strip(), authors, views.get(f"/{source[:-3]}/", 0))
        )
    return tuple(articles)
