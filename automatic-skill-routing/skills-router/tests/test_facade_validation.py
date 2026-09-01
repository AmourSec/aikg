from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runtime.contracts import (
    CandidateId,
    ExternalResponseToken,
    FacadeInvalidReason,
    NetworkConsent,
    SearchQuery,
    SkillName,
)
from runtime.coordinator import LocalExecutionMode
from runtime.facade import RouterTask, TaskRouterConfig
from runtime.token_registry import ResponseTokenRegistry
from runtime.wire import (
    PublicCandidates,
    PublicInvalid,
    WireLocalSelection,
    WireNetworkDecision,
    WireRemoteSelection,
    WireSelection,
    WireStart,
)
from tests.facade_support import (
    CatalogEntry,
    candidate_result,
    local_selection,
    make_task,
    remote_selection,
    write_catalog,
)
from tests.fakes import FakeProvider


class RouterTaskValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.catalog = write_catalog(
            self.workspace,
            (CatalogEntry("local", "skills/local/SKILL.md"),),
        )

    def _remote_task(
        self,
        handle: str = "valid-handle",
    ) -> tuple[RouterTask, ResponseTokenRegistry, FakeProvider, PublicCandidates]:
        provider = FakeProvider((candidate_result("remote-1", "remote-2"),), ())
        task, registry = make_task(
            self.workspace,
            self.catalog,
            provider,
            handle=handle,
        )
        task.start(WireStart(SearchQuery("remote"), (SkillName("local"),), False))
        outcome = task.resolve_network(WireNetworkDecision(NetworkConsent.GRANTED))
        assert isinstance(outcome, PublicCandidates)
        return task, registry, provider, outcome

    def test_forged_and_cross_task_handles_are_unknown_and_retryable(self) -> None:
        # Given
        task, _, provider, outcome = self._remote_task("task-a")
        other, _, _, other_outcome = self._remote_task("task-b")

        for handle in ("forged", str(other_outcome.response_token)):
            with self.subTest(handle=handle):
                # When
                invalid = task.select(remote_selection(handle, "remote-1"))

                # Then
                self.assertEqual(
                    invalid,
                    PublicInvalid(FacadeInvalidReason.UNKNOWN_RESPONSE_TOKEN),
                )
        valid = task.select(remote_selection(str(outcome.response_token), "remote-1"))
        self.assertNotIsInstance(valid, PublicInvalid)
        self.assertEqual(provider.load_requests, ())
        other.close()

    def test_duplicate_local_and_remote_selections_are_rejected(self) -> None:
        # Given
        task, _, _, outcome = self._remote_task()
        local = WireLocalSelection(SkillName("local"), LocalExecutionMode.NATIVE)
        duplicate_local = WireSelection((local, local), None)
        duplicate_remote = remote_selection(
            str(outcome.response_token),
            "remote-1",
            "remote-1",
        )

        # When / Then
        for selection in (duplicate_local, duplicate_remote):
            with self.subTest(selection=selection):
                self.assertEqual(
                    task.select(selection),
                    PublicInvalid(FacadeInvalidReason.DUPLICATE_SELECTION),
                )
        self.assertNotIsInstance(
            task.select(remote_selection(str(outcome.response_token), "remote-1")),
            PublicInvalid,
        )

    def test_unknown_local_and_mode_tampering_are_invalid_but_retryable(self) -> None:
        # Given
        provider = FakeProvider((), ())
        task, _ = make_task(self.workspace, self.catalog, provider)
        task.start(WireStart(SearchQuery("local"), (SkillName("local"),), True))
        cases = (
            WireLocalSelection(SkillName("unknown"), LocalExecutionMode.NATIVE),
            WireLocalSelection(SkillName("local"), LocalExecutionMode.PATH),
        )

        # When / Then
        for local in cases:
            with self.subTest(local=local):
                self.assertEqual(
                    task.select(WireSelection((local,), None)),
                    PublicInvalid(FacadeInvalidReason.INVALID_LOCAL_CANDIDATE),
                )
        self.assertNotIsInstance(task.select(local_selection()), PublicInvalid)

    def test_catalog_invalid_input_returns_public_invalid_without_transport(self) -> None:
        # Given
        self.catalog.write_text("not json", encoding="utf-8")
        provider = FakeProvider((), ())
        task, _ = make_task(self.workspace, self.catalog, provider)

        # When
        result = task.start(
            WireStart(SearchQuery("local"), (SkillName("local"),), False)
        )

        # Then
        self.assertEqual(
            result,
            PublicInvalid(FacadeInvalidReason.INVALID_LOCAL_CANDIDATE),
        )
        self.assertEqual(provider.search_requests, ())

    def test_duplicate_recalled_local_input_is_invalid_without_transport(self) -> None:
        # Given
        provider = FakeProvider((), ())
        task, _ = make_task(self.workspace, self.catalog, provider)

        # When
        result = task.start(
            WireStart(
                SearchQuery("local"),
                (SkillName("local"), SkillName("local")),
                False,
            )
        )

        # Then
        self.assertEqual(
            result,
            PublicInvalid(FacadeInvalidReason.INVALID_LOCAL_CANDIDATE),
        )
        self.assertEqual(provider.search_requests, ())

    def test_from_environment_creates_fresh_task_dependencies(self) -> None:
        # Given
        config = TaskRouterConfig(
            self.catalog,
            self.workspace,
            frozenset((SkillName("local"),)),
        )

        # When
        first = RouterTask.from_environment(config)
        second = RouterTask.from_environment(config)

        # Then
        self.assertIsNot(first._coordinator, second._coordinator)
        self.assertIsNot(first._coordinator.provider, second._coordinator.provider)
        self.assertIsNot(first._registry, second._registry)
        self.assertIsNot(
            first._coordinator.provider._transport,
            second._coordinator.provider._transport,
        )
        self.assertIsNot(
            first._coordinator.provider._sleeper,
            second._coordinator.provider._sleeper,
        )
        first.close()
        second.close()

    def test_invalid_transition_and_close_revoke_handle(self) -> None:
        # Given
        task, registry, _, outcome = self._remote_task()
        handle = outcome.response_token
        assert handle is not None

        # When
        before_start = make_task(
            self.workspace,
            self.catalog,
            FakeProvider((), ()),
        )[0].select(local_selection())
        task.close()

        # Then
        self.assertEqual(
            before_start,
            PublicInvalid(FacadeInvalidReason.INVALID_TRANSITION),
        )
        self.assertIsNone(registry.resolve(handle))
        self.assertEqual(
            task.select(
                WireSelection(
                    (),
                    WireRemoteSelection(
                        ExternalResponseToken(str(handle)),
                        outcome.remote_candidates[0].provider_id,
                        (CandidateId("remote-1"),),
                    ),
                )
            ),
            PublicInvalid(FacadeInvalidReason.INVALID_TRANSITION),
        )


if __name__ == "__main__":
    unittest.main()
