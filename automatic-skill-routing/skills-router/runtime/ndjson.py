from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Never, TypeAlias, assert_never

from runtime.contracts import (
    ActivationConsent,
    CandidateId,
    ExternalResponseToken,
    NetworkConsent,
    ProviderId,
    SearchQuery,
    SkillName,
)
from runtime.coordinator import LocalExecutionMode
from runtime.wire import (
    WireActivationDecision,
    WireLocalSelection,
    WireNetworkDecision,
    WireRemoteSelection,
    WireSelection,
    WireStart,
)

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@unique
class WireInvalidReason(StrEnum):
    BLANK_LINE = "blank_line"
    INVALID_JSON = "invalid_json"
    DUPLICATE_KEY = "duplicate_key"
    INVALID_SHAPE = "invalid_shape"
    INVALID_FIELDS = "invalid_fields"
    INVALID_ENUM = "invalid_enum"
    UNKNOWN_TYPE = "unknown_type"


@dataclass(frozen=True, slots=True)
class WireInvalid:
    reason: WireInvalidReason


@dataclass(frozen=True, slots=True)
class WireCancel:
    pass


DispatchMessage: TypeAlias = (
    WireStart
    | WireNetworkDecision
    | WireSelection
    | WireActivationDecision
)
WireMessage: TypeAlias = DispatchMessage | WireCancel
WireParseResult: TypeAlias = WireMessage | WireInvalid


@unique
class _MessageType(StrEnum):
    START = "start"
    NETWORK_DECISION = "network_decision"
    SELECTION = "selection"
    ACTIVATION_DECISION = "activation_decision"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class _DuplicateKeyError(ValueError):
    key: str

    def __str__(self) -> str:
        return f"duplicate JSON object key: {self.key}"


@dataclass(frozen=True, slots=True)
class _InvalidConstantError(ValueError):
    value: str

    def __str__(self) -> str:
        return f"non-standard JSON constant: {self.value}"


def parse_line(line: str) -> WireParseResult:
    if not line.strip():
        return WireInvalid(WireInvalidReason.BLANK_LINE)
    try:
        payload: JsonValue = json.loads(
            line,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except _DuplicateKeyError:
        return WireInvalid(WireInvalidReason.DUPLICATE_KEY)
    except (json.JSONDecodeError, _InvalidConstantError, RecursionError, ValueError):
        return WireInvalid(WireInvalidReason.INVALID_JSON)

    match payload:
        case dict() as fields:
            return _parse_message(fields)
        case list() | str() | int() | float() | bool() | None:
            return WireInvalid(WireInvalidReason.INVALID_SHAPE)
        case unreachable:
            assert_never(unreachable)


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> Never:
    raise _InvalidConstantError(value)


def _parse_message(fields: JsonObject) -> WireParseResult:
    raw_type = fields.get("type")
    match raw_type:
        case str() as type_name:
            try:
                message_type = _MessageType(type_name)
            except ValueError:
                return WireInvalid(WireInvalidReason.UNKNOWN_TYPE)
        case list() | dict() | int() | float() | bool() | None:
            return WireInvalid(WireInvalidReason.INVALID_FIELDS)
        case unreachable:
            assert_never(unreachable)

    match message_type:
        case _MessageType.START:
            return _parse_start(fields)
        case _MessageType.NETWORK_DECISION:
            return _parse_network(fields)
        case _MessageType.SELECTION:
            return _parse_selection(fields)
        case _MessageType.ACTIVATION_DECISION:
            return _parse_activation(fields)
        case _MessageType.CANCEL:
            if set(fields) != {"type"}:
                return WireInvalid(WireInvalidReason.INVALID_FIELDS)
            return WireCancel()
        case unreachable:
            assert_never(unreachable)


def _parse_start(fields: JsonObject) -> WireStart | WireInvalid:
    expected = {"type", "query", "recalled_local_names", "local_only"}
    if set(fields) != expected:
        return WireInvalid(WireInvalidReason.INVALID_FIELDS)
    query = fields["query"]
    names = fields["recalled_local_names"]
    local_only = fields["local_only"]
    if not isinstance(query, str) or not isinstance(names, list) or not isinstance(local_only, bool):
        return WireInvalid(WireInvalidReason.INVALID_FIELDS)
    parsed_names: list[SkillName] = []
    for raw_name in names:
        match raw_name:
            case str() as name:
                parsed_names.append(SkillName(name))
            case list() | dict() | int() | float() | bool() | None:
                return WireInvalid(WireInvalidReason.INVALID_FIELDS)
            case unreachable:
                assert_never(unreachable)
    return WireStart(SearchQuery(query), tuple(parsed_names), local_only)


def _parse_network(fields: JsonObject) -> WireNetworkDecision | WireInvalid:
    if set(fields) != {"type", "consent"}:
        return WireInvalid(WireInvalidReason.INVALID_FIELDS)
    match fields["consent"]:
        case str() as raw_consent:
            try:
                consent = NetworkConsent(raw_consent)
            except ValueError:
                return WireInvalid(WireInvalidReason.INVALID_ENUM)
            return WireNetworkDecision(consent)
        case list() | dict() | int() | float() | bool() | None:
            return WireInvalid(WireInvalidReason.INVALID_FIELDS)
        case unreachable:
            assert_never(unreachable)


def _parse_activation(fields: JsonObject) -> WireActivationDecision | WireInvalid:
    if set(fields) != {"type", "consent"}:
        return WireInvalid(WireInvalidReason.INVALID_FIELDS)
    match fields["consent"]:
        case str() as raw_consent:
            try:
                consent = ActivationConsent(raw_consent)
            except ValueError:
                return WireInvalid(WireInvalidReason.INVALID_ENUM)
            return WireActivationDecision(consent)
        case list() | dict() | int() | float() | bool() | None:
            return WireInvalid(WireInvalidReason.INVALID_FIELDS)
        case unreachable:
            assert_never(unreachable)


def _parse_selection(fields: JsonObject) -> WireSelection | WireInvalid:
    if set(fields) != {"type", "local", "remote"}:
        return WireInvalid(WireInvalidReason.INVALID_FIELDS)
    match fields["local"]:
        case list() as raw_local:
            local: list[WireLocalSelection] = []
            for raw_item in raw_local:
                item = _parse_local_selection(raw_item)
                match item:
                    case WireLocalSelection():
                        local.append(item)
                    case WireInvalid():
                        return item
                    case unreachable:
                        assert_never(unreachable)
        case dict() | str() | int() | float() | bool() | None:
            return WireInvalid(WireInvalidReason.INVALID_FIELDS)
        case unreachable:
            assert_never(unreachable)

    remote = _parse_remote_selection(fields["remote"])
    match remote:
        case WireRemoteSelection() | None:
            return WireSelection(tuple(local), remote)
        case WireInvalid():
            return remote
        case unreachable:
            assert_never(unreachable)


def _parse_local_selection(value: JsonValue) -> WireLocalSelection | WireInvalid:
    match value:
        case dict() as fields:
            if set(fields) != {"candidate_id", "execution_mode"}:
                return WireInvalid(WireInvalidReason.INVALID_FIELDS)
            candidate_id = fields["candidate_id"]
            execution_mode = fields["execution_mode"]
        case list() | str() | int() | float() | bool() | None:
            return WireInvalid(WireInvalidReason.INVALID_FIELDS)
        case unreachable:
            assert_never(unreachable)
    if not isinstance(candidate_id, str) or not isinstance(execution_mode, str):
        return WireInvalid(WireInvalidReason.INVALID_FIELDS)
    try:
        mode = LocalExecutionMode(execution_mode)
    except ValueError:
        return WireInvalid(WireInvalidReason.INVALID_ENUM)
    return WireLocalSelection(SkillName(candidate_id), mode)


def _parse_remote_selection(value: JsonValue) -> WireRemoteSelection | WireInvalid | None:
    match value:
        case None:
            return None
        case dict() as fields:
            expected = {"response_token", "provider_id", "candidate_ids"}
            if set(fields) != expected:
                return WireInvalid(WireInvalidReason.INVALID_FIELDS)
            token = fields["response_token"]
            provider = fields["provider_id"]
            candidate_ids = fields["candidate_ids"]
        case list() | str() | int() | float() | bool():
            return WireInvalid(WireInvalidReason.INVALID_FIELDS)
        case unreachable:
            assert_never(unreachable)
    if not isinstance(token, str) or not isinstance(provider, str) or not isinstance(candidate_ids, list):
        return WireInvalid(WireInvalidReason.INVALID_FIELDS)
    ids: list[CandidateId] = []
    for raw_id in candidate_ids:
        match raw_id:
            case str() as parsed_id:
                ids.append(CandidateId(parsed_id))
            case list() | dict() | int() | float() | bool() | None:
                return WireInvalid(WireInvalidReason.INVALID_FIELDS)
            case unreachable:
                assert_never(unreachable)
    return WireRemoteSelection(
        ExternalResponseToken(token),
        ProviderId(provider),
        tuple(ids),
    )
