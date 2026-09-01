from __future__ import annotations

import unittest

from runtime.contracts import CandidateId, ExternalResponseToken, ResponseToken
from runtime.token_registry import RegistryClosedError, ResponseTokenRegistry


class DeterministicIssuer:
    def __init__(self, handles: tuple[str, ...]) -> None:
        self._handles = iter(handles)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return next(self._handles)


class ResponseTokenRegistryTests(unittest.TestCase):
    def test_issue_resolves_the_exact_response_token_object(self) -> None:
        # Given
        token = ResponseToken(candidate_ids=(CandidateId("remote-1"),))
        registry = ResponseTokenRegistry(
            issuer=DeterministicIssuer(("deterministic-handle",)),
        )

        # When
        handle = registry.issue(token)

        # Then
        self.assertIs(registry.resolve(handle), token)

    def test_issue_uses_the_injected_zero_argument_issuer(self) -> None:
        # Given
        issuer = DeterministicIssuer(("known-handle",))
        registry = ResponseTokenRegistry(issuer=issuer)

        # When
        handle = registry.issue(ResponseToken())

        # Then
        self.assertEqual(handle, ExternalResponseToken("known-handle"))
        self.assertEqual(issuer.calls, 1)

    def test_forged_handle_does_not_resolve(self) -> None:
        # Given
        registry = ResponseTokenRegistry(
            issuer=DeterministicIssuer(("issued-handle",)),
        )
        registry.issue(ResponseToken())

        # When
        resolved = registry.resolve(ExternalResponseToken("forged-handle"))

        # Then
        self.assertIsNone(resolved)

    def test_handle_does_not_resolve_in_another_task_registry(self) -> None:
        # Given
        first_task = ResponseTokenRegistry(
            issuer=DeterministicIssuer(("first-task-handle",)),
        )
        second_task = ResponseTokenRegistry(
            issuer=DeterministicIssuer(("second-task-handle",)),
        )
        handle = first_task.issue(ResponseToken())

        # When
        resolved = second_task.resolve(handle)

        # Then
        self.assertIsNone(resolved)

    def test_close_revokes_issued_handles(self) -> None:
        # Given
        registry = ResponseTokenRegistry(
            issuer=DeterministicIssuer(("issued-handle",)),
        )
        handle = registry.issue(ResponseToken())

        # When
        registry.close()

        # Then
        self.assertIsNone(registry.resolve(handle))

    def test_issue_after_close_raises_registry_closed_error(self) -> None:
        # Given
        issuer = DeterministicIssuer(("issued-handle", "forbidden-handle"))
        registry = ResponseTokenRegistry(issuer=issuer)
        registry.issue(ResponseToken())
        registry.close()

        # When / Then
        with self.assertRaises(RegistryClosedError):
            registry.issue(ResponseToken())
        self.assertEqual(issuer.calls, 1)

    def test_production_handle_has_urlsafe_shape_without_candidate_ids(self) -> None:
        # Given
        candidate_id = CandidateId("private-remote-candidate-id")
        registry = ResponseTokenRegistry()

        # When
        handle = registry.issue(ResponseToken(candidate_ids=(candidate_id,)))

        # Then
        self.assertRegex(str(handle), r"^[A-Za-z0-9_-]{32}$")
        self.assertNotIn(str(candidate_id), str(handle))


if __name__ == "__main__":
    unittest.main()
