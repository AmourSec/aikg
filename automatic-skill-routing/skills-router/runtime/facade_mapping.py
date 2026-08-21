from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from runtime.contracts import (
    Ambiguous,
    Candidates,
    ExternalResponseToken,
    InlineRemoteTarget,
    InvalidResponse,
    LocalPathTarget,
    NativeLocalTarget,
    NoMatch,
    RemoteLoadInvalid,
    RemoteLoadUnavailable,
    RemoteCandidate,
    Unavailable,
)
from runtime.coordinator import ActivationRequired, DegradedStatus, ExecutionReady
from runtime.local_catalog import LocalCatalogDegradation, LocalCatalogResult
from runtime.rendering import (
    RemoteRenderInvalid,
    RenderedRemoteTarget,
    render_remote_target,
)
from runtime.wire import (
    PublicDegradation,
    PublicDegradationStage,
    PublicActivationRequired,
    PublicCandidates,
    PublicDegraded,
    PublicExecutionReady,
    PublicSearchDegradationReason,
)


def map_local_degradations(
    items: tuple[LocalCatalogDegradation, ...],
) -> tuple[PublicDegradation, ...]:
    return tuple(
        PublicDegradation(
            stage=PublicDegradationStage.LOCAL_CATALOG,
            reason=item.reason,
            candidate_id=item.candidate_id,
            path=item.path,
        )
        for item in items
    )


def map_degraded_statuses(
    items: tuple[DegradedStatus, ...],
) -> tuple[PublicDegradation, ...]:
    mapped: list[PublicDegradation] = []
    for item in items:
        match item:
            case NoMatch():
                mapped.append(
                    PublicDegradation(
                        PublicDegradationStage.SEARCH,
                        PublicSearchDegradationReason.NO_MATCH,
                    )
                )
            case Ambiguous():
                mapped.append(
                    PublicDegradation(
                        PublicDegradationStage.SEARCH,
                        PublicSearchDegradationReason.AMBIGUOUS,
                    )
                )
            case Unavailable(reason=reason) | InvalidResponse(reason=reason):
                mapped.append(
                    PublicDegradation(PublicDegradationStage.SEARCH, reason)
                )
            case RemoteLoadUnavailable(reason=reason) | RemoteLoadInvalid(reason=reason):
                mapped.append(
                    PublicDegradation(PublicDegradationStage.REMOTE_LOAD, reason)
                )
            case unreachable:
                assert_never(unreachable)
    return tuple(mapped)


@dataclass(frozen=True, slots=True)
class PublicCatalogView:
    local: LocalCatalogResult
    remote: tuple[RemoteCandidate, ...]
    degraded: tuple[PublicDegradation, ...]

    def candidates(
        self,
        handle: ExternalResponseToken | None,
    ) -> PublicCandidates:
        return PublicCandidates(
            self.local.candidates,
            self.local.modes,
            self.remote,
            handle,
            self.degraded,
        )

    def with_degradation(
        self,
        statuses: tuple[DegradedStatus, ...],
    ) -> PublicDegraded:
        return PublicDegraded(
            self.local.candidates,
            self.local.modes,
            (*self.degraded, *map_degraded_statuses(statuses)),
        )


def public_activation(
    state: ActivationRequired,
    initial: tuple[PublicDegradation, ...],
) -> PublicActivationRequired:
    selected = tuple(
        candidate
        for candidate_id in state.remote_ids
        for candidate in state.candidates.candidates
        if candidate.candidate_id == candidate_id
    )
    return PublicActivationRequired(state.local_targets, selected, initial)


def render_execution(
    state: ExecutionReady,
    candidates: Candidates | None,
    initial: tuple[PublicDegradation, ...],
) -> PublicExecutionReady:
    remote_by_id = (
        {}
        if candidates is None
        else {candidate.candidate_id: candidate for candidate in candidates.candidates}
    )
    targets: list[NativeLocalTarget | LocalPathTarget | RenderedRemoteTarget] = []
    degraded = [*initial, *map_degraded_statuses(state.degraded)]
    for target in state.targets:
        match target:
            case NativeLocalTarget() | LocalPathTarget():
                targets.append(target)
            case InlineRemoteTarget(content=content):
                rendered = render_remote_target(
                    remote_by_id[content.candidate_id],
                    content,
                )
                match rendered:
                    case RenderedRemoteTarget():
                        targets.append(rendered)
                    case RemoteRenderInvalid(candidate_id=candidate_id, reason=reason):
                        degraded.append(
                            PublicDegradation(
                                PublicDegradationStage.RENDER,
                                reason,
                                candidate_id,
                            )
                        )
                    case unreachable:
                        assert_never(unreachable)
            case unreachable:
                assert_never(unreachable)
    return PublicExecutionReady(tuple(targets), tuple(degraded))
