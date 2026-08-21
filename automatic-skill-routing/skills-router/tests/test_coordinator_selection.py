from __future__ import annotations

import unittest
from pathlib import Path

from runtime.contracts import (
    Ambiguous,
    CandidateId,
    Candidates,
    DisplayName,
    InvalidResponse,
    InvalidResponseReason,
    LocalCandidate,
    LocalPathTarget,
    LocalScore,
    NativeLocalTarget,
    ProviderId,
    RemoteCandidate,
    ResponseToken,
    SkillName,
    SourceFile,
    SourceRepo,
)
from runtime.coordinator import (
    ActivationRequired,
    CandidatesReady,
    Coordinator,
    DegradedReady,
    ExecutionReady,
    LocalExecutionMode,
    LocalOnlyReady,
    LocalSelectionRef,
    RemoteSelection,
    Selection,
    SelectionAmbiguous,
)
from tests.fakes import FakeProvider


def _local_candidate(name: str = "local-skill") -> LocalCandidate:
    return LocalCandidate(
        name=SkillName(name),
        description="local fallback",
        path=Path(f"skills/{name}/SKILL.md"),
        score=LocalScore(0.8),
    )


def _remote_candidate(candidate_id: str = "remote-1") -> RemoteCandidate:
    return RemoteCandidate(
        candidate_id=CandidateId(candidate_id),
        provider_id=ProviderId("provider"),
        display_name=DisplayName("remote skill"),
        source_repo=SourceRepo("org/repo"),
        source_file=SourceFile("skills/remote/SKILL.md"),
    )


class CoordinatorSelectionTests(unittest.TestCase):
    def test_local_selection_maps_installed_and_path_execution_modes(self) -> None:
        # Given
        installed = _local_candidate("installed")
        path_only = _local_candidate("path-only")
        state = LocalOnlyReady((installed, path_only))
        local_refs = (
            LocalSelectionRef(installed, LocalExecutionMode.NATIVE),
            LocalSelectionRef(path_only, LocalExecutionMode.PATH),
        )
        coordinator = Coordinator(FakeProvider((), ()))

        # When
        actual = coordinator.select(state, Selection(local_refs, None))

        # Then
        self.assertEqual(
            actual,
            ExecutionReady(
                (
                    NativeLocalTarget(installed.name),
                    LocalPathTarget(path_only.path),
                )
            ),
        )

    def test_unknown_local_candidate_yields_membership_degradation(self) -> None:
        # Given
        known = _local_candidate("known")
        unknown = _local_candidate("unknown")
        state = LocalOnlyReady((known,))
        coordinator = Coordinator(FakeProvider((), ()))

        # When
        actual = coordinator.select(
            state,
            Selection(
                (LocalSelectionRef(unknown, LocalExecutionMode.NATIVE),),
                None,
            ),
        )

        # Then
        self.assertEqual(
            actual,
            DegradedReady(
                (known,),
                (InvalidResponse(InvalidResponseReason.CANDIDATE_MEMBERSHIP),),
            ),
        )

    def test_remote_selection_requires_activation_without_load_call(self) -> None:
        # Given
        token = ResponseToken()
        remote = _remote_candidate()
        candidates = Candidates(token, (remote,))
        provider = FakeProvider((), ())
        coordinator = Coordinator(provider)
        state = CandidatesReady((_local_candidate(),), candidates)
        selection = RemoteSelection(
            token,
            remote.provider_id,
            (remote.candidate_id,),
        )

        # When
        actual = coordinator.select(state, Selection((), selection))

        # Then
        self.assertEqual(
            actual,
            ActivationRequired((), candidates, selection.candidate_ids),
        )
        self.assertEqual(provider.load_requests, ())

    def test_old_response_token_yields_candidate_membership_degradation(self) -> None:
        # Given
        token = ResponseToken()
        remote = _remote_candidate()
        candidates = Candidates(token, (remote,))
        local = _local_candidate()
        coordinator = Coordinator(FakeProvider((), ()))

        # When
        actual = coordinator.select(
            CandidatesReady((local,), candidates),
            Selection(
                (),
                RemoteSelection(
                    ResponseToken(),
                    remote.provider_id,
                    (remote.candidate_id,),
                ),
            ),
        )

        # Then
        self.assertEqual(
            actual,
            DegradedReady(
                (local,),
                (InvalidResponse(InvalidResponseReason.CANDIDATE_MEMBERSHIP),),
            ),
        )

    def test_unknown_remote_id_yields_candidate_membership_degradation(self) -> None:
        # Given
        token = ResponseToken()
        remote = _remote_candidate()
        candidates = Candidates(token, (remote,))
        local = _local_candidate()
        coordinator = Coordinator(FakeProvider((), ()))

        # When
        actual = coordinator.select(
            CandidatesReady((local,), candidates),
            Selection(
                (),
                RemoteSelection(
                    token,
                    remote.provider_id,
                    (CandidateId("unknown"),),
                ),
            ),
        )

        # Then
        self.assertEqual(
            actual,
            DegradedReady(
                (local,),
                (InvalidResponse(InvalidResponseReason.CANDIDATE_MEMBERSHIP),),
            ),
        )

    def test_explicit_ambiguity_rejects_remote_selection(self) -> None:
        # Given
        token = ResponseToken()
        remote = _remote_candidate()
        local = _local_candidate()
        state = SelectionAmbiguous(
            (local,),
            Ambiguous(token, (remote,)),
        )
        coordinator = Coordinator(FakeProvider((), ()))

        # When
        actual = coordinator.select(
            state,
            Selection(
                (),
                RemoteSelection(
                    token,
                    remote.provider_id,
                    (remote.candidate_id,),
                ),
            ),
        )

        # Then
        self.assertEqual(
            actual,
            DegradedReady(
                (local,),
                (InvalidResponse(InvalidResponseReason.CANDIDATE_MEMBERSHIP),),
            ),
        )

    def test_mixed_group_waits_as_a_whole_for_remote_activation(self) -> None:
        # Given
        local = _local_candidate()
        token = ResponseToken()
        remote = _remote_candidate()
        candidates = Candidates(token, (remote,))
        provider = FakeProvider((), ())
        coordinator = Coordinator(provider)

        # When
        actual = coordinator.select(
            CandidatesReady((local,), candidates),
            Selection(
                (LocalSelectionRef(local, LocalExecutionMode.NATIVE),),
                RemoteSelection(
                    token,
                    remote.provider_id,
                    (remote.candidate_id,),
                ),
            ),
        )

        # Then
        self.assertEqual(
            actual,
            ActivationRequired(
                (NativeLocalTarget(local.name),),
                candidates,
                (remote.candidate_id,),
            ),
        )
        self.assertNotIsInstance(actual, ExecutionReady)
        self.assertEqual(provider.load_requests, ())

    def test_mismatched_provider_id_yields_membership_without_load(self) -> None:
        # Given
        token = ResponseToken()
        remote = _remote_candidate()
        local = _local_candidate()
        candidates = Candidates(token, (remote,))
        provider = FakeProvider((), ())
        coordinator = Coordinator(provider)

        # When
        actual = coordinator.select(
            CandidatesReady((local,), candidates),
            Selection(
                (),
                RemoteSelection(
                    token,
                    ProviderId("other-provider"),
                    (remote.candidate_id,),
                ),
            ),
        )

        # Then
        self.assertEqual(
            actual,
            DegradedReady(
                (local,),
                (InvalidResponse(InvalidResponseReason.CANDIDATE_MEMBERSHIP),),
            ),
        )
        self.assertEqual(provider.load_requests, ())


if __name__ == "__main__":
    unittest.main()
