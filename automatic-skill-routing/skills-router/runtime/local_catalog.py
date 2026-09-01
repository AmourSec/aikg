from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import TypeAlias, TypeGuard, assert_never

from runtime.contracts import LocalCandidate, LocalScore, SkillName
from runtime.coordinator import LocalExecutionMode

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@unique
class LocalCatalogDegradationReason(StrEnum):
    MISSING = "missing"
    TRAVERSAL = "traversal"
    ABSOLUTE_ESCAPE = "absolute_escape"
    SYMLINK_ESCAPE = "symlink_escape"
    DIRECTORY = "directory"
    NOT_REGULAR_FILE = "not_regular_file"
    UNRESOLVABLE = "unresolvable"


@unique
class LocalCatalogInvalidRequestReason(StrEnum):
    DUPLICATE_RECALLED_ID = "duplicate_recalled_id"
    UNKNOWN_RECALLED_ID = "unknown_recalled_id"


@unique
class LocalCatalogParseReason(StrEnum):
    UNREADABLE = "unreadable"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"


@dataclass(frozen=True, slots=True)
class LocalCatalogRequest:
    recalled_ids: tuple[SkillName, ...]
    native_skill_names: frozenset[SkillName]


@dataclass(frozen=True, slots=True)
class LocalCatalogDegradation:
    candidate_id: SkillName
    path: Path
    reason: LocalCatalogDegradationReason


@dataclass(frozen=True, slots=True)
class LocalCatalogResult:
    candidates: tuple[LocalCandidate, ...]
    modes: tuple[LocalExecutionMode, ...]
    degraded: tuple[LocalCatalogDegradation, ...]

    @property
    def absent_cache(self) -> bool:
        return any(
            item.reason is LocalCatalogDegradationReason.MISSING
            and ".skills-cache" in item.path.parts
            for item in self.degraded
        )


@dataclass(frozen=True, slots=True)
class LocalCatalogInvalidRequestError(Exception):
    candidate_id: SkillName
    reason: LocalCatalogInvalidRequestReason

    def __str__(self) -> str:
        return f"invalid recalled local candidate {self.candidate_id!r}: {self.reason}"


@dataclass(frozen=True, slots=True)
class LocalCatalogParseError(Exception):
    path: Path
    reason: LocalCatalogParseReason

    def __str__(self) -> str:
        return f"invalid local catalog {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class _CatalogEntry:
    name: SkillName
    description: str
    path: Path


@dataclass(frozen=True, slots=True)
class _LoadablePath:
    path: Path


@dataclass(frozen=True, slots=True)
class _RejectedPath:
    path: Path
    reason: LocalCatalogDegradationReason


_PathAssessment: TypeAlias = _LoadablePath | _RejectedPath


def load_local_candidates(
    request: LocalCatalogRequest,
    catalog_path: Path,
    workspace_root: Path,
) -> LocalCatalogResult:
    seen: set[SkillName] = set()
    for candidate_id in request.recalled_ids:
        if candidate_id in seen:
            raise LocalCatalogInvalidRequestError(
                candidate_id,
                LocalCatalogInvalidRequestReason.DUPLICATE_RECALLED_ID,
            )
        seen.add(candidate_id)

    entries = _parse_catalog(catalog_path)
    root = workspace_root.resolve()
    candidates: list[LocalCandidate] = []
    modes: list[LocalExecutionMode] = []
    degraded: list[LocalCatalogDegradation] = []
    for candidate_id in request.recalled_ids:
        entry = entries.get(candidate_id)
        if entry is None:
            raise LocalCatalogInvalidRequestError(
                candidate_id,
                LocalCatalogInvalidRequestReason.UNKNOWN_RECALLED_ID,
            )
        if candidate_id in request.native_skill_names:
            candidate_path = entry.path if entry.path.is_absolute() else root / entry.path
            try:
                candidate_path = candidate_path.resolve()
            except OSError:
                candidate_path = candidate_path.absolute()
            candidates.append(_candidate(entry, candidate_path))
            modes.append(LocalExecutionMode.NATIVE)
            continue

        match _assess_path(entry.path, root):
            case _LoadablePath(path=resolved):
                candidates.append(_candidate(entry, resolved))
                modes.append(LocalExecutionMode.PATH)
            case _RejectedPath(path=path, reason=reason):
                degraded.append(
                    LocalCatalogDegradation(candidate_id, path, reason)
                )
            case unreachable:
                assert_never(unreachable)
    return LocalCatalogResult(tuple(candidates), tuple(modes), tuple(degraded))


def _candidate(entry: _CatalogEntry, path: Path) -> LocalCandidate:
    return LocalCandidate(entry.name, entry.description, path, LocalScore(0.0))


def _assess_path(path: Path, root: Path) -> _PathAssessment:
    candidate = path if path.is_absolute() else root / path
    if not path.is_absolute() and ".." in path.parts:
        return _RejectedPath(
            candidate.resolve(),
            LocalCatalogDegradationReason.TRAVERSAL,
        )
    try:
        resolved = candidate.resolve()
    except OSError:
        return _RejectedPath(
            candidate.absolute(),
            LocalCatalogDegradationReason.UNRESOLVABLE,
        )
    if not resolved.is_relative_to(root):
        reason = (
            LocalCatalogDegradationReason.ABSOLUTE_ESCAPE
            if path.is_absolute()
            else LocalCatalogDegradationReason.SYMLINK_ESCAPE
        )
        return _RejectedPath(resolved, reason)
    if not resolved.exists():
        return _RejectedPath(resolved, LocalCatalogDegradationReason.MISSING)
    if resolved.is_dir():
        return _RejectedPath(resolved, LocalCatalogDegradationReason.DIRECTORY)
    if not resolved.is_file():
        return _RejectedPath(
            resolved,
            LocalCatalogDegradationReason.NOT_REGULAR_FILE,
        )
    return _LoadablePath(resolved)


def _parse_catalog(catalog_path: Path) -> dict[SkillName, _CatalogEntry]:
    try:
        body = catalog_path.read_bytes()
    except OSError as error:
        raise LocalCatalogParseError(
            catalog_path,
            LocalCatalogParseReason.UNREADABLE,
        ) from error
    try:
        payload: JsonValue = json.loads(body)
    except (RecursionError, ValueError) as error:
        raise LocalCatalogParseError(
            catalog_path,
            LocalCatalogParseReason.INVALID_JSON,
        ) from error
    match payload:
        case dict() as top_level:
            raw_skills = top_level.get("skills")
        case list() | str() | int() | float() | bool() | None:
            raise LocalCatalogParseError(
                catalog_path,
                LocalCatalogParseReason.INVALID_SCHEMA,
            )
        case unreachable:
            assert_never(unreachable)
    match raw_skills:
        case list() as raw_entries:
            pass
        case dict() | str() | int() | float() | bool() | None:
            raise LocalCatalogParseError(
                catalog_path,
                LocalCatalogParseReason.INVALID_SCHEMA,
            )
        case unreachable:
            assert_never(unreachable)

    entries: dict[SkillName, _CatalogEntry] = {}
    for raw_entry in raw_entries:
        entry = _parse_entry(raw_entry, catalog_path)
        if entry.name in entries:
            raise LocalCatalogParseError(
                catalog_path,
                LocalCatalogParseReason.INVALID_SCHEMA,
            )
        entries[entry.name] = entry
    return entries


def _parse_entry(raw: JsonValue, catalog_path: Path) -> _CatalogEntry:
    match raw:
        case dict() as fields:
            name = fields.get("name")
            description = fields.get("description")
            path = fields.get("path")
        case list() | str() | int() | float() | bool() | None:
            raise LocalCatalogParseError(
                catalog_path,
                LocalCatalogParseReason.INVALID_SCHEMA,
            )
        case unreachable:
            assert_never(unreachable)
    if not (
        _is_identifier(name)
        and _is_description(description)
        and _is_identifier(path)
    ):
        raise LocalCatalogParseError(
            catalog_path,
            LocalCatalogParseReason.INVALID_SCHEMA,
        )
    return _CatalogEntry(SkillName(name), description, Path(path))


def _is_identifier(value: JsonValue) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and value == value.strip()
        and bool(value)
        and value.isprintable()
    )


def _is_description(value: JsonValue) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip())
