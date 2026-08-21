from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Literal, NewType, Protocol, TypeAlias, assert_never

SkillName = NewType("SkillName", str)
CandidateId = NewType("CandidateId", str)
ExternalResponseToken = NewType("ExternalResponseToken", str)
ProviderId = NewType("ProviderId", str)
DisplayName = NewType("DisplayName", str)
SourceRepo = NewType("SourceRepo", str)
SourceFile = NewType("SourceFile", str)
SearchQuery = NewType("SearchQuery", str)
UntrustedText = NewType("UntrustedText", str)
Endpoint = NewType("Endpoint", str)
HeaderName = NewType("HeaderName", str)
HeaderValue = NewType("HeaderValue", str)
ByteLimit = NewType("ByteLimit", int)
LocalScore = NewType("LocalScore", float)
ProviderScore = NewType("ProviderScore", str)
Seconds = NewType("Seconds", float)

Header: TypeAlias = tuple[HeaderName, HeaderValue]
Headers: TypeAlias = tuple[Header, ...]


@unique
class NetworkConsent(StrEnum):
    NOT_REQUESTED = "not_requested"
    GRANTED = "granted"
    REFUSED = "refused"


@unique
class ActivationConsent(StrEnum):
    NOT_REQUESTED = "not_requested"
    GRANTED = "granted"
    REFUSED = "refused"


@unique
class UnavailableReason(StrEnum):
    NO_API_KEY = "no_api_key"
    CONFIGURATION = "configuration"
    RATE_LIMITED = "rate_limited"
    SERVICE = "service"
    TIMEOUT = "timeout"


@unique
class InvalidResponseReason(StrEnum):
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"
    OVERSIZED = "oversized"
    CANDIDATE_MEMBERSHIP = "candidate_membership"
    CONSENT_REQUIRED = "consent_required"


@unique
class FacadeInvalidReason(StrEnum):
    UNKNOWN_RESPONSE_TOKEN = "unknown_response_token"
    DUPLICATE_SELECTION = "duplicate_selection"
    DELIMITER_COLLISION = "delimiter_collision"
    INVALID_TRANSITION = "invalid_transition"
    INVALID_LOCAL_CANDIDATE = "invalid_local_candidate"


@unique
class ContentTrust(StrEnum):
    UNTRUSTED_EXTERNAL = "untrusted_external"


@unique
class HttpMethod(StrEnum):
    GET = "GET"
    POST = "POST"


@dataclass(frozen=True, slots=True)
class LocalCandidate:
    name: SkillName
    description: str
    path: Path
    score: LocalScore


@dataclass(frozen=True, slots=True)
class RemoteCandidate:
    candidate_id: CandidateId
    provider_id: ProviderId
    display_name: DisplayName
    source_repo: SourceRepo
    source_file: SourceFile
    score: ProviderScore | None = None
    trust: ContentTrust = ContentTrust.UNTRUSTED_EXTERNAL
    policy_authority: Literal[False] = False


@dataclass(frozen=True, slots=True, eq=False)
class ResponseToken:
    candidate_ids: tuple[CandidateId, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedSearch:
    query: SearchQuery
    body: bytes


def prepare_search(query: SearchQuery) -> PreparedSearch:
    payload = {
        "query": str(query),
        "top_k": 10,
        "with_neighbors": False,
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return PreparedSearch(query=query, body=body)


@dataclass(frozen=True, slots=True, init=False)
class SearchRequest:
    prepared: PreparedSearch
    network_consent: NetworkConsent

    def __init__(
        self,
        prepared: PreparedSearch | SearchQuery,
        network_consent: NetworkConsent,
    ) -> None:
        match prepared:
            case PreparedSearch() as normalized:
                pass
            case str() as legacy_query:
                normalized = prepare_search(SearchQuery(legacy_query))
            case unreachable:
                assert_never(unreachable)
        object.__setattr__(self, "prepared", normalized)
        object.__setattr__(self, "network_consent", network_consent)

    @property
    def query(self) -> SearchQuery:
        return self.prepared.query


@dataclass(frozen=True, slots=True)
class LoadRequest:
    response_token: ResponseToken
    candidate_id: CandidateId
    network_consent: NetworkConsent
    activation_consent: ActivationConsent


@dataclass(frozen=True, slots=True)
class Candidates:
    response_token: ResponseToken
    candidates: tuple[RemoteCandidate, ...]

    def contains(self, request: LoadRequest) -> bool:
        return request.response_token is self.response_token and any(
            candidate.candidate_id == request.candidate_id
            for candidate in self.candidates
        )


@dataclass(frozen=True, slots=True)
class NoMatch:
    pass


@dataclass(frozen=True, slots=True)
class Ambiguous:
    response_token: ResponseToken
    candidates: tuple[RemoteCandidate, ...]


@dataclass(frozen=True, slots=True)
class Unavailable:
    reason: UnavailableReason


@dataclass(frozen=True, slots=True)
class InvalidResponse:
    reason: InvalidResponseReason


ProviderResult: TypeAlias = (
    Candidates | NoMatch | Ambiguous | Unavailable | InvalidResponse
)
SearchResult: TypeAlias = ProviderResult


@dataclass(frozen=True, slots=True)
class RemoteSkillContent:
    response_token: ResponseToken
    candidate_id: CandidateId
    content: UntrustedText
    trust: ContentTrust = ContentTrust.UNTRUSTED_EXTERNAL
    policy_authority: Literal[False] = False


@dataclass(frozen=True, slots=True)
class RemoteLoadUnavailable:
    reason: UnavailableReason


@dataclass(frozen=True, slots=True)
class RemoteLoadInvalid:
    reason: InvalidResponseReason


RemoteLoadResult: TypeAlias = (
    RemoteSkillContent | RemoteLoadUnavailable | RemoteLoadInvalid
)


@dataclass(frozen=True, slots=True)
class NativeLocalTarget:
    skill_name: SkillName


@dataclass(frozen=True, slots=True)
class LocalPathTarget:
    path: Path


@dataclass(frozen=True, slots=True)
class InlineRemoteTarget:
    content: RemoteSkillContent


ExecutionTarget: TypeAlias = (
    NativeLocalTarget | LocalPathTarget | InlineRemoteTarget
)


@dataclass(frozen=True, slots=True)
class TransportRequest:
    method: HttpMethod
    endpoint: Endpoint
    headers: Headers
    body: bytes
    timeout_seconds: Seconds
    max_response_bytes: ByteLimit


@dataclass(frozen=True, slots=True)
class TransportResponse:
    status_code: int
    headers: Headers
    body: bytes


@dataclass(frozen=True, slots=True)
class TransportTimeout:
    timeout_seconds: Seconds


@dataclass(frozen=True, slots=True)
class TransportUnavailable:
    reason: UnavailableReason


TransportResult: TypeAlias = (
    TransportResponse | TransportTimeout | TransportUnavailable
)


class Provider(Protocol):
    def search(self, request: SearchRequest) -> SearchResult: ...

    def load_skill(self, request: LoadRequest) -> RemoteLoadResult: ...


class Transport(Protocol):
    def send(
        self,
        request: TransportRequest,
    ) -> TransportResult: ...


class Sleeper(Protocol):
    def sleep(self, delay: Seconds) -> None: ...
