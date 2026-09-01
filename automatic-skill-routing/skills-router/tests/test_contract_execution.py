from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import assert_never

from runtime.contracts import (
    CandidateId,
    ContentTrust,
    ExecutionTarget,
    InlineRemoteTarget,
    InvalidResponseReason,
    LocalPathTarget,
    NativeLocalTarget,
    RemoteLoadInvalid,
    RemoteLoadResult,
    RemoteLoadUnavailable,
    RemoteSkillContent,
    ResponseToken,
    SkillName,
    UnavailableReason,
    UntrustedText,
)


def _remote_content() -> RemoteSkillContent:
    return RemoteSkillContent(
        response_token=ResponseToken(),
        candidate_id=CandidateId("remote-1"),
        content=UntrustedText("remote skill body"),
    )


def _remote_load_result_name(result: RemoteLoadResult) -> str:
    match result:
        case RemoteSkillContent():
            return "content"
        case RemoteLoadUnavailable():
            return "unavailable"
        case RemoteLoadInvalid():
            return "invalid"
        case unreachable:
            assert_never(unreachable)


def _execution_target_name(target: ExecutionTarget) -> str:
    match target:
        case NativeLocalTarget():
            return "native_local"
        case LocalPathTarget():
            return "local_path"
        case InlineRemoteTarget():
            return "inline_remote"
        case unreachable:
            assert_never(unreachable)


class RemoteLoadTests(unittest.TestCase):
    def test_remote_content_is_untrusted_and_has_no_policy_authority(self) -> None:
        # Given
        content = _remote_content()

        # When
        policy = (content.trust, content.policy_authority)

        # Then
        self.assertEqual(policy, (ContentTrust.UNTRUSTED_EXTERNAL, False))
        with self.assertRaises(FrozenInstanceError):
            setattr(content, "policy_authority", True)

    def test_remote_load_outcomes_are_separate_exhaustive_variants(self) -> None:
        # Given
        outcomes: tuple[RemoteLoadResult, ...] = (
            _remote_content(),
            RemoteLoadUnavailable(reason=UnavailableReason.TIMEOUT),
            RemoteLoadInvalid(reason=InvalidResponseReason.CANDIDATE_MEMBERSHIP),
        )

        # When
        names = tuple(_remote_load_result_name(outcome) for outcome in outcomes)

        # Then
        self.assertEqual(names, ("content", "unavailable", "invalid"))


class ExecutionTargetTests(unittest.TestCase):
    def test_execution_targets_keep_native_path_and_remote_content_separate(self) -> None:
        # Given
        native = NativeLocalTarget(skill_name=SkillName("local-skill"))
        local_path = LocalPathTarget(path=Path("skills/local-skill/SKILL.md"))
        remote_content = _remote_content()
        inline_remote = InlineRemoteTarget(content=remote_content)
        targets: tuple[ExecutionTarget, ...] = (
            native,
            local_path,
            inline_remote,
        )

        # When
        names = tuple(_execution_target_name(target) for target in targets)

        # Then
        self.assertEqual(names, ("native_local", "local_path", "inline_remote"))
        self.assertEqual(native.skill_name, SkillName("local-skill"))
        self.assertEqual(
            local_path.path,
            Path("skills/local-skill/SKILL.md"),
        )
        self.assertIs(inline_remote.content, remote_content)


if __name__ == "__main__":
    unittest.main()
