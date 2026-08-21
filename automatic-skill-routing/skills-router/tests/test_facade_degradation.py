from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runtime.contracts import (
    Ambiguous,
    CandidateId,
    InvalidResponse,
    InvalidResponseReason,
    NetworkConsent,
    NoMatch,
    ResponseToken,
    SearchQuery,
    SkillName,
    Unavailable,
    UnavailableReason,
)
from runtime.local_catalog import LocalCatalogDegradationReason
from runtime.wire import (
    PublicCandidates,
    PublicDegradationStage,
    PublicDegraded,
    PublicSearchDegradationReason,
    WireNetworkDecision,
    WireStart,
)
from tests.facade_support import (
    CatalogEntry,
    make_task,
    remote_candidate,
    write_catalog,
)
from tests.fakes import FakeProvider


class RouterTaskDegradationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_missing_cache_is_typed_local_degradation(self) -> None:
        # Given
        catalog = write_catalog(
            self.workspace,
            (CatalogEntry("cached", ".skills-cache/missing/SKILL.md"),),
        )
        task, _ = make_task(
            self.workspace,
            catalog,
            FakeProvider((), ()),
            native_names=(),
        )

        # When
        result = task.start(
            WireStart(SearchQuery("local"), (SkillName("cached"),), True)
        )

        # Then
        self.assertIsInstance(result, PublicCandidates)
        assert isinstance(result, PublicCandidates)
        self.assertEqual(result.local_candidates, ())
        self.assertEqual(result.degraded[0].stage, PublicDegradationStage.LOCAL_CATALOG)
        self.assertEqual(result.degraded[0].reason, LocalCatalogDegradationReason.MISSING)

    def test_all_non_candidate_search_results_are_typed_and_selectable(self) -> None:
        token = ResponseToken((CandidateId("remote-1"),))
        cases = (
            (NoMatch(), PublicSearchDegradationReason.NO_MATCH),
            (
                Ambiguous(token, (remote_candidate(),)),
                PublicSearchDegradationReason.AMBIGUOUS,
            ),
            (Unavailable(UnavailableReason.SERVICE), UnavailableReason.SERVICE),
            (
                InvalidResponse(InvalidResponseReason.INVALID_SCHEMA),
                InvalidResponseReason.INVALID_SCHEMA,
            ),
        )
        for index, (search_result, reason) in enumerate(cases):
            with self.subTest(reason=reason):
                catalog = write_catalog(
                    self.workspace,
                    (CatalogEntry("local", "skills/local/SKILL.md"),),
                )
                task, _ = make_task(
                    self.workspace,
                    catalog,
                    FakeProvider((search_result,), ()),
                    handle=f"unused-{index}",
                )
                task.start(
                    WireStart(SearchQuery("remote"), (SkillName("local"),), False)
                )

                # When
                result = task.resolve_network(
                    WireNetworkDecision(NetworkConsent.GRANTED)
                )

                # Then
                self.assertIsInstance(result, PublicDegraded)
                assert isinstance(result, PublicDegraded)
                self.assertEqual(result.local_candidates[0].name, SkillName("local"))
                self.assertEqual(result.degraded[-1].reason, reason)


if __name__ == "__main__":
    unittest.main()
