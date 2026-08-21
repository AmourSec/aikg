from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypeAlias, assert_never

from runtime.contracts import (
    LocalCandidate,
    LocalPathTarget,
    NativeLocalTarget,
    RemoteCandidate,
)
from runtime.ndjson import JsonValue, WireInvalid
from runtime.rendering import RenderedRemoteTarget
from runtime.wire import (
    FacadeOutcome,
    PublicActivationRequired,
    PublicCandidates,
    PublicDegradation,
    PublicDegraded,
    PublicExecutionReady,
    PublicExecutionTarget,
    PublicInvalid,
    SearchDisclosure,
)


@dataclass(frozen=True, slots=True)
class Cancelled:
    pass


HostOutcome: TypeAlias = FacadeOutcome | WireInvalid | Cancelled
LocalTarget: TypeAlias = NativeLocalTarget | LocalPathTarget


def encode_outcome(outcome: HostOutcome) -> str:
    return json.dumps(
        _outcome_value(outcome),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _outcome_value(outcome: HostOutcome) -> JsonValue:
    match outcome:
        case SearchDisclosure():
            return {
                "type": "search_disclosure",
                "endpoint": str(outcome.endpoint),
                "method": outcome.method.value,
                "body": outcome.body,
                "local_candidates": [_local_candidate(item) for item in outcome.local_candidates],
                "local_modes": [mode.value for mode in outcome.local_modes],
                "degraded": [_degradation(item) for item in outcome.degraded],
            }
        case PublicCandidates():
            return {
                "type": "candidates",
                "local_candidates": [_local_candidate(item) for item in outcome.local_candidates],
                "local_modes": [mode.value for mode in outcome.local_modes],
                "remote_candidates": [_remote_candidate(item) for item in outcome.remote_candidates],
                "response_token": None if outcome.response_token is None else str(outcome.response_token),
                "degraded": [_degradation(item) for item in outcome.degraded],
            }
        case PublicActivationRequired():
            return {
                "type": "activation_required",
                "local_targets": [_local_target(item) for item in outcome.local_targets],
                "remote_candidates": [_remote_candidate(item) for item in outcome.remote_candidates],
                "degraded": [_degradation(item) for item in outcome.degraded],
            }
        case PublicExecutionReady():
            return {
                "type": "execution_ready",
                "targets": [_execution_target(item) for item in outcome.targets],
                "degraded": [_degradation(item) for item in outcome.degraded],
            }
        case PublicDegraded():
            return {
                "type": "degraded",
                "local_candidates": [_local_candidate(item) for item in outcome.local_candidates],
                "local_modes": [mode.value for mode in outcome.local_modes],
                "degraded": [_degradation(item) for item in outcome.degraded],
            }
        case PublicInvalid():
            return {"type": "invalid", "reason": outcome.reason.value}
        case WireInvalid():
            return {"type": "wire_invalid", "reason": outcome.reason.value}
        case Cancelled():
            return {"type": "cancelled"}
        case unreachable:
            assert_never(unreachable)


def _local_candidate(candidate: LocalCandidate) -> JsonValue:
    return {
        "name": str(candidate.name),
        "description": candidate.description,
        "path": str(candidate.path),
        "score": float(candidate.score),
    }


def _remote_candidate(candidate: RemoteCandidate) -> JsonValue:
    return {
        "candidate_id": str(candidate.candidate_id),
        "provider_id": str(candidate.provider_id),
        "display_name": str(candidate.display_name),
        "source_repo": str(candidate.source_repo),
        "source_file": str(candidate.source_file),
        "score": None if candidate.score is None else str(candidate.score),
        "trust": candidate.trust.value,
        "policy_authority": candidate.policy_authority,
    }


def _degradation(degradation: PublicDegradation) -> JsonValue:
    return {
        "stage": degradation.stage.value,
        "reason": degradation.reason.value,
        "candidate_id": (
            None if degradation.candidate_id is None else str(degradation.candidate_id)
        ),
        "path": None if degradation.path is None else str(degradation.path),
    }


def _local_target(target: LocalTarget) -> JsonValue:
    match target:
        case NativeLocalTarget():
            return {"type": "native_local", "skill_name": str(target.skill_name)}
        case LocalPathTarget():
            return {"type": "local_path", "path": str(target.path)}
        case unreachable:
            assert_never(unreachable)


def _execution_target(target: PublicExecutionTarget) -> JsonValue:
    match target:
        case NativeLocalTarget() | LocalPathTarget():
            return _local_target(target)
        case RenderedRemoteTarget():
            return {
                "type": "rendered_remote",
                "candidate_id": str(target.candidate_id),
                "source_file": str(target.source_file),
                "rendered": target.rendered,
            }
        case unreachable:
            assert_never(unreachable)
