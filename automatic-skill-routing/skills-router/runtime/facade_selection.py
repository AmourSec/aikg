from __future__ import annotations

from dataclasses import dataclass

from runtime.contracts import FacadeInvalidReason, RemoteCandidate
from runtime.coordinator import LocalSelectionRef, RemoteSelection, Selection
from runtime.local_catalog import LocalCatalogResult
from runtime.token_registry import ResponseTokenRegistry
from runtime.wire import PublicInvalid, WireSelection


@dataclass(frozen=True, slots=True)
class SelectionValidator:
    local: LocalCatalogResult
    remote_candidates: tuple[RemoteCandidate, ...]
    registry: ResponseTokenRegistry

    def build(self, request: WireSelection) -> Selection | PublicInvalid:
        local_ids = tuple(item.candidate_id for item in request.local)
        remote_ids = () if request.remote is None else request.remote.candidate_ids
        if len(set(local_ids)) != len(local_ids) or len(set(remote_ids)) != len(remote_ids):
            return PublicInvalid(FacadeInvalidReason.DUPLICATE_SELECTION)
        local_by_id = {
            candidate.name: (candidate, mode)
            for candidate, mode in zip(self.local.candidates, self.local.modes)
        }
        local_refs: list[LocalSelectionRef] = []
        for item in request.local:
            known = local_by_id.get(item.candidate_id)
            if known is None or known[1] is not item.execution_mode:
                return PublicInvalid(FacadeInvalidReason.INVALID_LOCAL_CANDIDATE)
            local_refs.append(LocalSelectionRef(known[0], known[1]))
        wire = request.remote
        if wire is None:
            return Selection(tuple(local_refs), None)
        token = self.registry.resolve(wire.response_token)
        if token is None or any(
            candidate_id not in token.candidate_ids
            for candidate_id in wire.candidate_ids
        ):
            return PublicInvalid(FacadeInvalidReason.UNKNOWN_RESPONSE_TOKEN)
        selected = tuple(
            candidate
            for candidate in self.remote_candidates
            if candidate.candidate_id in wire.candidate_ids
        )
        if len(selected) != len(wire.candidate_ids) or any(
            candidate.provider_id != wire.provider_id for candidate in selected
        ):
            return PublicInvalid(FacadeInvalidReason.UNKNOWN_RESPONSE_TOKEN)
        return Selection(
            tuple(local_refs),
            RemoteSelection(token, wire.provider_id, wire.candidate_ids),
        )
