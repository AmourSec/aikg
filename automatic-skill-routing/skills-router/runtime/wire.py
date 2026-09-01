from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import TypeAlias

from runtime.contracts import (
    ActivationConsent,
    CandidateId,
    Endpoint,
    ExternalResponseToken,
    FacadeInvalidReason,
    HttpMethod,
    InvalidResponseReason,
    LocalCandidate,
    LocalPathTarget,
    NativeLocalTarget,
    NetworkConsent,
    ProviderId,
    RemoteCandidate,
    SearchQuery,
    SkillName,
    UnavailableReason,
)
from runtime.coordinator import LocalExecutionMode
from runtime.local_catalog import LocalCatalogDegradationReason
from runtime.rendering import RenderedRemoteTarget


@dataclass(frozen=True, slots=True)
class WireStart:
    query: SearchQuery
    recalled_local_names: tuple[SkillName, ...]
    local_only: bool


@dataclass(frozen=True, slots=True)
class WireNetworkDecision:
    consent: NetworkConsent


@dataclass(frozen=True, slots=True)
class WireLocalSelection:
    candidate_id: SkillName
    execution_mode: LocalExecutionMode


@dataclass(frozen=True, slots=True)
class WireRemoteSelection:
    response_token: ExternalResponseToken
    provider_id: ProviderId
    candidate_ids: tuple[CandidateId, ...]


@dataclass(frozen=True, slots=True)
class WireSelection:
    local: tuple[WireLocalSelection, ...]
    remote: WireRemoteSelection | None


@dataclass(frozen=True, slots=True)
class WireActivationDecision:
    consent: ActivationConsent


@unique
class PublicDegradationStage(StrEnum):
    LOCAL_CATALOG = "local_catalog"
    SEARCH = "search"
    REMOTE_LOAD = "remote_load"
    RENDER = "render"


@unique
class PublicSearchDegradationReason(StrEnum):
    NO_MATCH = "no_match"
    AMBIGUOUS = "ambiguous"


PublicDegradationReason: TypeAlias = (
    LocalCatalogDegradationReason
    | PublicSearchDegradationReason
    | UnavailableReason
    | InvalidResponseReason
    | FacadeInvalidReason
)


@dataclass(frozen=True, slots=True)
class PublicDegradation:
    stage: PublicDegradationStage
    reason: PublicDegradationReason
    candidate_id: SkillName | CandidateId | None = None
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class SearchDisclosure:
    endpoint: Endpoint
    method: HttpMethod
    body: str
    local_candidates: tuple[LocalCandidate, ...]
    local_modes: tuple[LocalExecutionMode, ...]
    degraded: tuple[PublicDegradation, ...]


@dataclass(frozen=True, slots=True)
class PublicCandidates:
    local_candidates: tuple[LocalCandidate, ...]
    local_modes: tuple[LocalExecutionMode, ...]
    remote_candidates: tuple[RemoteCandidate, ...]
    response_token: ExternalResponseToken | None
    degraded: tuple[PublicDegradation, ...]


@dataclass(frozen=True, slots=True)
class PublicActivationRequired:
    local_targets: tuple[NativeLocalTarget | LocalPathTarget, ...]
    remote_candidates: tuple[RemoteCandidate, ...]
    degraded: tuple[PublicDegradation, ...]


PublicExecutionTarget: TypeAlias = (
    NativeLocalTarget | LocalPathTarget | RenderedRemoteTarget
)


@dataclass(frozen=True, slots=True)
class PublicExecutionReady:
    targets: tuple[PublicExecutionTarget, ...]
    degraded: tuple[PublicDegradation, ...]


@dataclass(frozen=True, slots=True)
class PublicDegraded:
    local_candidates: tuple[LocalCandidate, ...]
    local_modes: tuple[LocalExecutionMode, ...]
    degraded: tuple[PublicDegradation, ...]


@dataclass(frozen=True, slots=True)
class PublicInvalid:
    reason: FacadeInvalidReason


FacadeOutcome: TypeAlias = (
    SearchDisclosure
    | PublicCandidates
    | PublicActivationRequired
    | PublicExecutionReady
    | PublicDegraded
    | PublicInvalid
)
