from __future__ import annotations

import json
import math
from typing import Final, TypeAlias, TypeGuard, assert_never

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from runtime.contracts import (
    ByteLimit, CandidateId, Candidates, DisplayName, InvalidResponse,
    InvalidResponseReason, LoadRequest, NoMatch, ProviderId, ProviderScore,
    RemoteCandidate, RemoteLoadInvalid, RemoteLoadResult, RemoteSkillContent,
    ResponseToken, SearchResult, SourceFile, SourceRepo, UntrustedText,
)

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_PROVIDER_ID: Final = ProviderId("ascend-kg")
_SEARCH_LIMIT: Final = ByteLimit(1_048_576)
_SKILL_LIMIT: Final = ByteLimit(262_144)
_MAX_CANDIDATE_ID_CHARS: Final = 512
_MAX_SOURCE_REPO_CHARS: Final = 1_024
_MAX_SOURCE_FILE_CHARS: Final = 1_024
_MAX_DISPLAY_NAME_CHARS: Final = 256
_MAX_PROVIDER_SCORE_CHARS: Final = 128
_REQUIRED_FRONTMATTER_KEYS: Final = frozenset(("name", "description"))


def parse_search_response(body: bytes) -> SearchResult:
    if len(body) > _SEARCH_LIMIT:
        return InvalidResponse(InvalidResponseReason.OVERSIZED)
    try:
        payload: JsonValue = json.loads(body)
    except (UnicodeDecodeError, RecursionError, ValueError):
        return InvalidResponse(InvalidResponseReason.INVALID_JSON)
    match payload:
        case dict() as top_level:
            has_results = "results" in top_level
            has_data = "data" in top_level
            if has_results == has_data:
                return InvalidResponse(InvalidResponseReason.INVALID_SCHEMA)
            entries = top_level["results"] if has_results else top_level["data"]
        case list() | str() | int() | float() | bool() | None:
            return InvalidResponse(InvalidResponseReason.INVALID_SCHEMA)
        case unreachable:
            assert_never(unreachable)
    match entries:
        case list() as raw_entries:
            if len(raw_entries) > 10:
                return InvalidResponse(InvalidResponseReason.INVALID_SCHEMA)
        case dict() | str() | int() | float() | bool() | None:
            return InvalidResponse(InvalidResponseReason.INVALID_SCHEMA)
        case unreachable:
            assert_never(unreachable)
    candidates: list[RemoteCandidate] = []
    seen: set[CandidateId] = set()
    for raw_entry in raw_entries:
        candidate = _parse_candidate(raw_entry)
        if candidate is None or candidate.candidate_id in seen:
            return InvalidResponse(InvalidResponseReason.INVALID_SCHEMA)
        candidates.append(candidate)
        seen.add(candidate.candidate_id)
    if not candidates:
        return NoMatch()
    candidate_tuple = tuple(candidates)
    token = ResponseToken(
        candidate_ids=tuple(candidate.candidate_id for candidate in candidate_tuple),
    )
    return Candidates(response_token=token, candidates=candidate_tuple)


def _parse_candidate(raw: JsonValue) -> RemoteCandidate | None:
    match raw:
        case dict() as entry:
            has_candidate_id = "id" in entry
            has_legacy_id = "node_id" in entry
            if not has_candidate_id and not has_legacy_id:
                return None
            candidate_id = entry["id"] if has_candidate_id else entry["node_id"]
            if has_candidate_id and has_legacy_id and candidate_id != entry["node_id"]:
                return None
            source_repo = entry.get("source_repo")
            source_file = entry.get("source_file")
            score = entry.get("score")
        case list() | str() | int() | float() | bool() | None:
            return None
        case unreachable:
            assert_never(unreachable)
    if not _is_safe_text(candidate_id, _MAX_CANDIDATE_ID_CHARS):
        return None
    if not _is_safe_text(source_repo, _MAX_SOURCE_REPO_CHARS):
        return None
    if not _is_safe_text(source_file, _MAX_SOURCE_FILE_CHARS):
        return None
    path_parts = tuple(part for part in source_file.split("/") if part)
    if not path_parts:
        return None
    if path_parts[-1] == "SKILL.md":
        if len(path_parts) < 2:
            return None
        display_name = path_parts[-2]
    else:
        display_name = path_parts[-1]
    if not _is_safe_text(display_name, _MAX_DISPLAY_NAME_CHARS):
        return None
    match score:
        case None:
            provider_score = None
        case bool() | list() | dict():
            return None
        case str():
            if not _is_safe_text(score, _MAX_PROVIDER_SCORE_CHARS):
                return None
            provider_score = ProviderScore(score)
        case int():
            rendered_score = str(score)
            if not _is_safe_text(rendered_score, _MAX_PROVIDER_SCORE_CHARS):
                return None
            provider_score = ProviderScore(rendered_score)
        case float():
            if not math.isfinite(score):
                return None
            rendered_score = str(score)
            if not _is_safe_text(rendered_score, _MAX_PROVIDER_SCORE_CHARS):
                return None
            provider_score = ProviderScore(rendered_score)
        case unreachable:
            assert_never(unreachable)
    return RemoteCandidate(
        candidate_id=CandidateId(candidate_id),
        provider_id=_PROVIDER_ID,
        display_name=DisplayName(display_name),
        source_repo=SourceRepo(source_repo),
        source_file=SourceFile(source_file),
        score=provider_score,
    )


def _is_safe_text(value: JsonValue, max_chars: int) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and value == value.strip()
        and 0 < len(value) <= max_chars
        and value.isprintable()
    )


def parse_skill_response(request: LoadRequest, body: bytes) -> RemoteLoadResult:
    if len(body) > _SKILL_LIMIT:
        return RemoteLoadInvalid(InvalidResponseReason.OVERSIZED)
    try:
        content = body.decode("utf-8")
    except UnicodeDecodeError:
        return RemoteLoadInvalid(InvalidResponseReason.INVALID_SCHEMA)
    lines = content.splitlines()
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        return RemoteLoadInvalid(InvalidResponseReason.INVALID_SCHEMA)
    closing = lines.index("---", 1)
    has_body = any(line.strip() for line in lines[closing + 1 :])
    frontmatter = "\n".join(lines[1:closing])
    if not has_body or not _is_valid_frontmatter(frontmatter):
        return RemoteLoadInvalid(InvalidResponseReason.INVALID_SCHEMA)
    return RemoteSkillContent(
        response_token=request.response_token,
        candidate_id=request.candidate_id,
        content=UntrustedText(content),
    )


def _is_valid_frontmatter(frontmatter: str) -> bool:
    try:
        root = yaml.compose(frontmatter, Loader=yaml.SafeLoader)
        yaml.safe_load(frontmatter)
    except (RecursionError, ValueError, yaml.YAMLError):
        return False
    if not isinstance(root, MappingNode):
        return False

    required_values: dict[str, list[Node]] = {
        key: [] for key in _REQUIRED_FRONTMATTER_KEYS
    }
    for key_node, value_node in root.value:
        if (
            isinstance(key_node, ScalarNode)
            and key_node.value in _REQUIRED_FRONTMATTER_KEYS
        ):
            if key_node.start_mark.column != 0:
                return False
            required_values[key_node.value].append(value_node)

    if any(len(values) != 1 for values in required_values.values()):
        return False
    if any(
        not isinstance(values[0], ScalarNode)
        or values[0].tag != "tag:yaml.org,2002:str"
        or not values[0].value.strip()
        for values in required_values.values()
    ):
        return False
    return not _contains_required_key(
        tuple(value_node for _, value_node in root.value),
    )


def _contains_required_key(nodes: tuple[Node, ...]) -> bool:
    pending = list(nodes)
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        node_identity = id(node)
        if node_identity in seen:
            continue
        seen.add(node_identity)
        match node:
            case MappingNode(value=pairs):
                for key_node, value_node in pairs:
                    if (
                        isinstance(key_node, ScalarNode)
                        and key_node.value in _REQUIRED_FRONTMATTER_KEYS
                    ):
                        return True
                    pending.extend((key_node, value_node))
            case SequenceNode(value=children):
                pending.extend(children)
            case ScalarNode():
                pass
            case unreachable:
                assert_never(unreachable)
    return False
