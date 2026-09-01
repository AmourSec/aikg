from __future__ import annotations

import unittest
from pathlib import Path

from runtime.contracts import (
    CandidateId,
    Candidates,
    DisplayName,
    InvalidResponse,
    InvalidResponseReason,
    LocalCandidate,
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
    LocalExecutionMode,
    LocalOnlyReady,
    LocalSelectionRef,
    RemoteSelection,
    Selection,
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


def _membership_invalid() -> InvalidResponse:
    return InvalidResponse(InvalidResponseReason.CANDIDATE_MEMBERSHIP)


class CoordinatorDuplicateSelectionTests(unittest.TestCase):
    def test_repeated_local_name_rejects_whole_group_without_targets(self) -> None:
        # Given
        local = _local_candidate()
        token = ResponseToken()
        remote = _remote_candidate()
        candidates = Candidates(token, (remote,))
        provider = FakeProvider((), ())
        coordinator = Coordinator(provider)
        selection = Selection(
            (
                LocalSelectionRef(local, LocalExecutionMode.NATIVE),
                LocalSelectionRef(local, LocalExecutionMode.PATH),
            ),
            RemoteSelection(token, remote.provider_id, (remote.candidate_id,)),
        )

        # When
        actual = coordinator.select(CandidatesReady((local,), candidates), selection)

        # Then
        self.assertEqual(
            actual,
            DegradedReady((local,), (_membership_invalid(),)),
        )
        self.assertNotIsInstance(actual, ActivationRequired)
        self.assertEqual(provider.load_requests, ())

    def test_two_candidate_objects_sharing_skill_name_are_rejected(self) -> None:
        # Given
        first = _local_candidate("shared-name")
        second = LocalCandidate(
            name=SkillName("shared-name"),
            description="different description",
            path=Path("skills/other-place/SKILL.md"),
            score=LocalScore(0.5),
        )
        provider = FakeProvider((), ())
        coordinator = Coordinator(provider)
        selection = Selection(
            (
                LocalSelectionRef(first, LocalExecutionMode.NATIVE),
                LocalSelectionRef(second, LocalExecutionMode.PATH),
            ),
            None,
        )

        # When
        actual = coordinator.select(LocalOnlyReady((first, second)), selection)

        # Then
        self.assertEqual(
            actual,
            DegradedReady((first, second), (_membership_invalid(),)),
        )
        self.assertEqual(provider.load_requests, ())

    def test_repeated_remote_candidate_id_rejects_whole_group_without_targets(self) -> None:
        # Given
        local = _local_candidate()
        token = ResponseToken()
        first = _remote_candidate("remote-1")
        second = _remote_candidate("remote-2")
        candidates = Candidates(token, (first, second))
        provider = FakeProvider((), ())
        coordinator = Coordinator(provider)
        selection = Selection(
            (LocalSelectionRef(local, LocalExecutionMode.NATIVE),),
            RemoteSelection(
                token,
                first.provider_id,
                (first.candidate_id, second.candidate_id, first.candidate_id),
            ),
        )

        # When
        actual = coordinator.select(CandidatesReady((local,), candidates), selection)

        # Then
        self.assertEqual(
            actual,
            DegradedReady((local,), (_membership_invalid(),)),
        )
        self.assertNotIsInstance(actual, ActivationRequired)
        self.assertEqual(provider.load_requests, ())

    def test_local_names_and_remote_ids_form_independent_namespaces(self) -> None:
        # Given
        local = _local_candidate("shared-id")
        token = ResponseToken()
        remote = _remote_candidate("shared-id")
        candidates = Candidates(token, (remote,))
        provider = FakeProvider((), ())
        coordinator = Coordinator(provider)

        # When
        actual = coordinator.select(
            CandidatesReady((local,), candidates),
            Selection(
                (LocalSelectionRef(local, LocalExecutionMode.NATIVE),),
                RemoteSelection(token, remote.provider_id, (remote.candidate_id,)),
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
        self.assertEqual(provider.load_requests, ())


if __name__ == "__main__":
    unittest.main()
