from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TypeAlias, assert_never

from runtime.contracts import (
    ActivationConsent,
    Ambiguous,
    CandidateId,
    Candidates,
    ExecutionTarget,
    InlineRemoteTarget,
    InvalidResponse,
    InvalidResponseReason,
    LoadRequest,
    LocalCandidate,
    LocalPathTarget,
    NativeLocalTarget,
    NetworkConsent,
    NoMatch,
    PreparedSearch, Provider, ProviderId,
    RemoteLoadInvalid,
    RemoteLoadUnavailable,
    RemoteSkillContent,
    ResponseToken,
    SearchQuery,
    SearchRequest,
    Unavailable, prepare_search,
)

LocalTarget: TypeAlias = NativeLocalTarget | LocalPathTarget
DegradedStatus: TypeAlias = (
    NoMatch | Ambiguous | Unavailable | InvalidResponse
    | RemoteLoadUnavailable | RemoteLoadInvalid
)


@unique
class LocalExecutionMode(StrEnum):
    NATIVE = "native"
    PATH = "path"


@dataclass(frozen=True, slots=True)
class LocalSelectionRef:
    candidate: LocalCandidate
    execution_mode: LocalExecutionMode


@dataclass(frozen=True, slots=True)
class RemoteSelection:
    response_token: ResponseToken
    provider_id: ProviderId
    candidate_ids: tuple[CandidateId, ...]


@dataclass(frozen=True, slots=True)
class StartRequest:
    query: SearchQuery
    local_only: bool


@dataclass(frozen=True, slots=True)
class NetworkDecision:
    consent: NetworkConsent


@dataclass(frozen=True, slots=True)
class Selection:
    local_references: tuple[LocalSelectionRef, ...]
    remote: RemoteSelection | None


@dataclass(frozen=True, slots=True)
class LocalOnlyReady:
    local_candidates: tuple[LocalCandidate, ...]
    network_consent: NetworkConsent = NetworkConsent.NOT_REQUESTED


@dataclass(frozen=True, slots=True)
class NetworkConsentRequired:
    local_candidates: tuple[LocalCandidate, ...]
    prepared: PreparedSearch


@dataclass(frozen=True, slots=True)
class CandidatesReady:
    local_candidates: tuple[LocalCandidate, ...]
    candidates: Candidates


@dataclass(frozen=True, slots=True)
class SelectionNoMatch:
    local_candidates: tuple[LocalCandidate, ...]
    degraded: NoMatch


@dataclass(frozen=True, slots=True)
class SelectionAmbiguous:
    local_candidates: tuple[LocalCandidate, ...]
    degraded: Ambiguous


@dataclass(frozen=True, slots=True)
class DegradedReady:
    local_candidates: tuple[LocalCandidate, ...]
    degraded: tuple[DegradedStatus, ...]


@dataclass(frozen=True, slots=True)
class ActivationRequired:
    local_targets: tuple[LocalTarget, ...]
    candidates: Candidates
    remote_ids: tuple[CandidateId, ...]


@dataclass(frozen=True, slots=True)
class ActivationRefused:
    targets: tuple[LocalTarget, ...]


@dataclass(frozen=True, slots=True)
class ExecutionReady:
    targets: tuple[ExecutionTarget, ...]
    degraded: tuple[DegradedStatus, ...] = ()


StartState: TypeAlias = LocalOnlyReady | NetworkConsentRequired
NetworkResolved: TypeAlias = StartState | CandidatesReady | SelectionNoMatch | SelectionAmbiguous | DegradedReady
SelectionState: TypeAlias = LocalOnlyReady | CandidatesReady | SelectionNoMatch | SelectionAmbiguous | DegradedReady
SelectionResult: TypeAlias = ActivationRequired | ExecutionReady | DegradedReady
ActivationResult: TypeAlias = ActivationRequired | ActivationRefused | ExecutionReady


@dataclass(frozen=True, slots=True)
class Coordinator:
    provider: Provider

    def start(
        self,
        local_candidates: tuple[LocalCandidate, ...],
        request: StartRequest,
    ) -> StartState:
        if request.local_only:
            return LocalOnlyReady(local_candidates)
        return NetworkConsentRequired(local_candidates, prepare_search(request.query))

    def resolve_network(
        self,
        state: NetworkConsentRequired,
        decision: NetworkDecision,
    ) -> NetworkResolved:
        match decision.consent:
            case NetworkConsent.NOT_REQUESTED:
                return state
            case NetworkConsent.REFUSED:
                return LocalOnlyReady(state.local_candidates, decision.consent)
            case NetworkConsent.GRANTED:
                result = self.provider.search(
                    SearchRequest(state.prepared, decision.consent)
                )
            case unreachable:
                assert_never(unreachable)

        match result:
            case Candidates():
                return CandidatesReady(state.local_candidates, result)
            case NoMatch():
                return SelectionNoMatch(state.local_candidates, result)
            case Ambiguous():
                return SelectionAmbiguous(state.local_candidates, result)
            case Unavailable() | InvalidResponse():
                return DegradedReady(state.local_candidates, (result,))
            case unreachable:
                assert_never(unreachable)

    def select(
        self,
        state: SelectionState,
        selection: Selection,
    ) -> SelectionResult:
        match state:
            case LocalOnlyReady(local_candidates=local_candidates):
                candidates = None
                degraded: tuple[DegradedStatus, ...] = ()
            case CandidatesReady(
                local_candidates=local_candidates,
                candidates=candidates,
            ):
                degraded = ()
            case SelectionNoMatch(
                local_candidates=local_candidates,
                degraded=status,
            ):
                candidates = None
                degraded = (status,)
            case SelectionAmbiguous(
                local_candidates=local_candidates,
                degraded=status,
            ):
                candidates = None
                degraded = (status,)
            case DegradedReady(
                local_candidates=local_candidates,
                degraded=degraded,
            ):
                candidates = None
            case unreachable:
                assert_never(unreachable)

        invalid = InvalidResponse(InvalidResponseReason.CANDIDATE_MEMBERSHIP)
        refs = selection.local_references
        if any(
            ref.candidate not in local_candidates for ref in refs
        ) or len({ref.candidate.name for ref in refs}) != len(refs):
            return DegradedReady(local_candidates, (invalid,))

        remote = selection.remote
        remote_ids = () if remote is None else remote.candidate_ids
        if remote_ids and (
            candidates is None
            or remote is None
            or remote.response_token is not candidates.response_token
            or len(set(remote_ids)) != len(remote_ids)
            or any(
                candidate_id
                not in tuple(candidate.candidate_id for candidate in candidates.candidates)
                for candidate_id in remote_ids
            )
            or any(
                candidate.provider_id != remote.provider_id
                for candidate in candidates.candidates
                if candidate.candidate_id in remote_ids
            )
        ):
            return DegradedReady(local_candidates, (invalid,))

        local_targets: list[LocalTarget] = []
        for reference in selection.local_references:
            match reference.execution_mode:
                case LocalExecutionMode.NATIVE:
                    local_targets.append(NativeLocalTarget(reference.candidate.name))
                case LocalExecutionMode.PATH:
                    local_targets.append(LocalPathTarget(reference.candidate.path))
                case unreachable:
                    assert_never(unreachable)

        if remote_ids:
            assert candidates is not None
            return ActivationRequired(tuple(local_targets), candidates, remote_ids)
        return ExecutionReady(tuple(local_targets), degraded)

    def activate(
        self,
        state: ActivationRequired,
        consent: ActivationConsent,
    ) -> ActivationResult:
        match consent:
            case ActivationConsent.NOT_REQUESTED:
                return state
            case ActivationConsent.REFUSED:
                return ActivationRefused(state.local_targets)
            case ActivationConsent.GRANTED:
                pass
            case unreachable:
                assert_never(unreachable)

        targets: list[ExecutionTarget] = list(state.local_targets)
        degraded: list[DegradedStatus] = []
        for candidate_id in state.remote_ids:
            request = LoadRequest(
                state.candidates.response_token,
                candidate_id,
                NetworkConsent.GRANTED,
                consent,
            )
            result = self.provider.load_skill(request)
            match result:
                case RemoteSkillContent():
                    if (
                        result.response_token is state.candidates.response_token
                        and result.candidate_id == candidate_id
                    ):
                        targets.append(InlineRemoteTarget(result))
                    else:
                        degraded.append(
                            RemoteLoadInvalid(
                                InvalidResponseReason.CANDIDATE_MEMBERSHIP
                            )
                        )
                case RemoteLoadUnavailable() | RemoteLoadInvalid():
                    degraded.append(result)
                case unreachable:
                    assert_never(unreachable)
        return ExecutionReady(tuple(targets), tuple(degraded))
