from __future__ import annotations

import unittest

from runtime.contracts import NetworkConsent, NoMatch, SearchQuery
from runtime.coordinator import (
    Coordinator,
    NetworkConsentRequired,
    NetworkDecision,
    StartRequest,
)
from tests.fakes import FakeProvider


class CoordinatorPreparedSearchTests(unittest.TestCase):
    def test_start_prepares_exact_body_before_network_consent(self) -> None:
        # Given
        coordinator = Coordinator(FakeProvider((), ()))

        # When
        state = coordinator.start(
            (),
            StartRequest(SearchQuery("查找 NPU skill"), local_only=False),
        )

        # Then
        self.assertIsInstance(state, NetworkConsentRequired)
        assert isinstance(state, NetworkConsentRequired)
        self.assertEqual(
            state.prepared.body,
            b'{"query":"\xe6\x9f\xa5\xe6\x89\xbe NPU skill","top_k":10,"with_neighbors":false}',
        )

    def test_grant_passes_disclosed_prepared_object_to_provider(self) -> None:
        # Given
        provider = FakeProvider((NoMatch(),), ())
        coordinator = Coordinator(provider)
        state = coordinator.start(
            (),
            StartRequest(SearchQuery("find a skill"), local_only=False),
        )
        assert isinstance(state, NetworkConsentRequired)

        # When
        coordinator.resolve_network(
            state,
            NetworkDecision(NetworkConsent.GRANTED),
        )

        # Then
        self.assertIs(provider.search_requests[0].prepared, state.prepared)


if __name__ == "__main__":
    unittest.main()
