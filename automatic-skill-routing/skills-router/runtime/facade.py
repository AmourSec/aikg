from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import assert_never

from runtime.ascend_kg import AscendKgProvider, ProductionSleeper, SEARCH_ENDPOINT
from runtime.contracts import (
    Candidates,
    FacadeInvalidReason,
    HttpMethod,
    RemoteCandidate,
    SkillName,
)
from runtime.coordinator import (
    ActivationRefused,
    ActivationRequired,
    CandidatesReady,
    Coordinator, DegradedReady, DegradedStatus,
    ExecutionReady,
    LocalOnlyReady,
    NetworkConsentRequired,
    NetworkDecision,
    Selection,
    SelectionAmbiguous,
    SelectionNoMatch,
    SelectionState,
    StartRequest,
)
from runtime.facade_mapping import (
    map_local_degradations,
    PublicCatalogView,
    public_activation,
    render_execution,
)
from runtime.facade_selection import SelectionValidator
from runtime.http_transport import UrllibTransport
from runtime.local_catalog import (
    LocalCatalogInvalidRequestError,
    LocalCatalogParseError,
    LocalCatalogRequest,
    LocalCatalogResult,
    load_local_candidates,
)
from runtime.token_registry import ResponseTokenRegistry
from runtime.wire import (
    FacadeOutcome,
    PublicDegradation,
    PublicDegraded,
    PublicInvalid,
    SearchDisclosure,
    WireActivationDecision,
    WireNetworkDecision,
    WireSelection,
    WireStart,
)

@dataclass(frozen=True, slots=True)
class TaskRouterConfig:
    catalog_path: Path
    workspace_root: Path
    native_skill_names: frozenset[SkillName]


@unique
class _Phase(StrEnum):
    NEW = "new"
    NETWORK = "network"
    SELECTION = "selection"
    ACTIVATION = "activation"
    CLOSED = "closed"


class RouterTask:
    __slots__ = (
        "_activation_state",
        "_config",
        "_coordinator",
        "_disclosure",
        "_initial_degraded",
        "_local_result",
        "_network_state",
        "_phase",
        "_registry",
        "_remote_candidates",
        "_selection_state",
    )

    def __init__(
        self,
        config: TaskRouterConfig,
        coordinator: Coordinator,
        registry: ResponseTokenRegistry,
    ) -> None:
        self._config = config
        self._coordinator = coordinator
        self._registry = registry
        self._phase = _Phase.NEW
        self._local_result: LocalCatalogResult | None = None
        self._network_state: NetworkConsentRequired | None = None
        self._selection_state: SelectionState | None = None
        self._activation_state: ActivationRequired | None = None
        self._disclosure: SearchDisclosure | None = None
        self._remote_candidates: tuple[RemoteCandidate, ...] = ()
        self._initial_degraded: tuple[PublicDegradation, ...] = ()

    @classmethod
    def from_environment(cls, config: TaskRouterConfig) -> RouterTask:
        provider = AscendKgProvider.from_environment(
            UrllibTransport(),
            ProductionSleeper(),
        )
        return cls(config, Coordinator(provider), ResponseTokenRegistry())

    def start(self, request: WireStart) -> FacadeOutcome:
        if self._phase is not _Phase.NEW:
            return PublicInvalid(FacadeInvalidReason.INVALID_TRANSITION)
        try:
            local_result = load_local_candidates(
                LocalCatalogRequest(
                    request.recalled_local_names,
                    self._config.native_skill_names,
                ),
                self._config.catalog_path,
                self._config.workspace_root,
            )
        except (LocalCatalogInvalidRequestError, LocalCatalogParseError):
            self.close()
            return PublicInvalid(FacadeInvalidReason.INVALID_LOCAL_CANDIDATE)
        self._local_result = local_result
        self._initial_degraded = map_local_degradations(local_result.degraded)
        state = self._coordinator.start(
            local_result.candidates,
            StartRequest(request.query, request.local_only),
        )
        match state:
            case LocalOnlyReady():
                self._phase = _Phase.SELECTION
                self._selection_state = state
                return self._catalog_view().candidates(None)
            case NetworkConsentRequired(prepared=prepared):
                self._phase = _Phase.NETWORK
                self._network_state = state
                self._disclosure = SearchDisclosure(
                    SEARCH_ENDPOINT,
                    HttpMethod.POST,
                    prepared.body.decode("utf-8"),
                    local_result.candidates,
                    local_result.modes,
                    self._initial_degraded,
                )
                return self._disclosure
            case unreachable:
                assert_never(unreachable)

    def resolve_network(self, decision: WireNetworkDecision) -> FacadeOutcome:
        state = self._network_state
        if self._phase is not _Phase.NETWORK or state is None:
            return PublicInvalid(FacadeInvalidReason.INVALID_TRANSITION)
        result = self._coordinator.resolve_network(
            state,
            NetworkDecision(decision.consent),
        )
        match result:
            case NetworkConsentRequired():
                assert self._disclosure is not None
                return self._disclosure
            case LocalOnlyReady():
                self._selection_state = result
                self._phase = _Phase.SELECTION
                return self._catalog_view().candidates(None)
            case CandidatesReady(candidates=candidates):
                self._selection_state = result
                self._remote_candidates = candidates.candidates
                self._phase = _Phase.SELECTION
                return self._catalog_view().candidates(
                    self._registry.issue(candidates.response_token)
                )
            case SelectionNoMatch(degraded=degraded):
                self._selection_state = result
                return self._degraded_outcome((degraded,))
            case SelectionAmbiguous(degraded=degraded):
                self._selection_state = result
                return self._degraded_outcome((degraded,))
            case DegradedReady(degraded=degraded):
                self._selection_state = result
                return self._degraded_outcome(degraded)
            case unreachable:
                assert_never(unreachable)

    def select(self, request: WireSelection) -> FacadeOutcome:
        state = self._selection_state
        local_result = self._local_result
        if self._phase is not _Phase.SELECTION or state is None or local_result is None:
            return PublicInvalid(FacadeInvalidReason.INVALID_TRANSITION)
        selection = SelectionValidator(
            local_result,
            self._remote_candidates,
            self._registry,
        ).build(request)
        match selection:
            case PublicInvalid():
                return selection
            case Selection():
                result = self._coordinator.select(state, selection)
            case unreachable:
                assert_never(unreachable)
        match result:
            case ExecutionReady():
                return self._complete(result, None)
            case ActivationRequired():
                self._activation_state = result
                self._phase = _Phase.ACTIVATION
                return public_activation(result, self._initial_degraded)
            case DegradedReady():
                return PublicInvalid(FacadeInvalidReason.UNKNOWN_RESPONSE_TOKEN)
            case unreachable:
                assert_never(unreachable)

    def activate(self, decision: WireActivationDecision) -> FacadeOutcome:
        state = self._activation_state
        if self._phase is not _Phase.ACTIVATION or state is None:
            return PublicInvalid(FacadeInvalidReason.INVALID_TRANSITION)
        result = self._coordinator.activate(state, decision.consent)
        match result:
            case ActivationRequired():
                return public_activation(result, self._initial_degraded)
            case ActivationRefused(targets=targets):
                return self._complete(ExecutionReady(targets), state.candidates)
            case ExecutionReady():
                return self._complete(result, state.candidates)
            case unreachable:
                assert_never(unreachable)

    def close(self) -> None:
        self._registry.close()
        self._phase = _Phase.CLOSED

    def _degraded_outcome(
        self,
        degraded: tuple[DegradedStatus, ...],
    ) -> PublicDegraded:
        assert self._local_result is not None
        self._phase = _Phase.SELECTION
        return self._catalog_view().with_degradation(degraded)

    def _catalog_view(self) -> PublicCatalogView:
        assert self._local_result is not None
        return PublicCatalogView(
            self._local_result,
            self._remote_candidates,
            self._initial_degraded,
        )

    def _complete(
        self,
        state: ExecutionReady,
        candidates: Candidates | None,
    ) -> FacadeOutcome:
        result = render_execution(state, candidates, self._initial_degraded)
        self.close()
        return result
