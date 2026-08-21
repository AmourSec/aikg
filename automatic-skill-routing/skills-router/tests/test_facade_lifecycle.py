from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runtime.contracts import (
    ActivationConsent,
    CandidateId,
    FacadeInvalidReason,
    HttpMethod,
    InvalidResponse,
    InvalidResponseReason,
    NativeLocalTarget,
    NetworkConsent,
    NoMatch,
    RemoteLoadUnavailable,
    RemoteSkillContent,
    SearchQuery,
    SkillName,
    Unavailable,
    UnavailableReason,
    UntrustedText,
)
from runtime.rendering import RenderedRemoteTarget
from runtime.wire import (
    PublicActivationRequired,
    PublicCandidates,
    PublicDegradationStage,
    PublicExecutionReady,
    PublicInvalid,
    PublicSearchDegradationReason,
    SearchDisclosure,
    WireActivationDecision,
    WireNetworkDecision,
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


class RouterTaskLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.catalog = write_catalog(
            self.workspace,
            (CatalogEntry("local", "skills/local/SKILL.md"),),
        )

    def test_local_only_selection_completes_and_closes(self) -> None:
        # Given
        provider = FakeProvider((), ())
        task, _ = make_task(self.workspace, self.catalog, provider)
        start = WireStart(SearchQuery("local task"), (SkillName("local"),), True)

        # When
        candidates = task.start(start)
        result = task.select(local_selection())

        # Then
        self.assertIsInstance(candidates, PublicCandidates)
        self.assertEqual(
            result,
            PublicExecutionReady((NativeLocalTarget(SkillName("local")),), ()),
        )
        self.assertEqual(provider.search_requests, ())
        self.assertEqual(
            task.select(local_selection()),
            PublicInvalid(FacadeInvalidReason.INVALID_TRANSITION),
        )

    def test_network_disclosure_is_exact_and_refusal_stays_local(self) -> None:
        # Given
        provider = FakeProvider((), ())
        task, _ = make_task(self.workspace, self.catalog, provider)

        # When
        disclosure = task.start(
            WireStart(SearchQuery("查找 skill"), (SkillName("local"),), False)
        )
        pending = task.resolve_network(
            WireNetworkDecision(NetworkConsent.NOT_REQUESTED)
        )
        refused = task.resolve_network(WireNetworkDecision(NetworkConsent.REFUSED))

        # Then
        self.assertIsInstance(disclosure, SearchDisclosure)
        assert isinstance(disclosure, SearchDisclosure)
        self.assertEqual(disclosure.endpoint, "https://ascend.wiki/search")
        self.assertEqual(disclosure.method, HttpMethod.POST)
        self.assertEqual(
            disclosure.body,
            '{"query":"查找 skill","top_k":10,"with_neighbors":false}',
        )
        self.assertEqual(pending, disclosure)
        self.assertIsInstance(refused, PublicCandidates)
        self.assertEqual(provider.search_requests, ())

    def test_remote_candidates_use_one_opaque_handle_and_wait_for_activation(self) -> None:
        # Given
        remote = candidate_result("remote-1")
        provider = FakeProvider((remote,), ())
        task, _ = make_task(self.workspace, self.catalog, provider)
        task.start(WireStart(SearchQuery("remote"), (SkillName("local"),), False))

        # When
        candidates = task.resolve_network(WireNetworkDecision(NetworkConsent.GRANTED))
        assert isinstance(candidates, PublicCandidates)
        selected = task.select(
            remote_selection(str(candidates.response_token), "remote-1", include_local=True)
        )

        # Then
        self.assertEqual(candidates.response_token, "opaque-task-handle")
        self.assertNotIn("remote-1", str(candidates.response_token))
        self.assertIsInstance(selected, PublicActivationRequired)
        self.assertEqual(provider.load_requests, ())

    def test_activation_not_requested_then_refused_preserves_local_target(self) -> None:
        # Given
        remote = candidate_result("remote-1")
        provider = FakeProvider((remote,), ())
        task, _ = make_task(self.workspace, self.catalog, provider)
        task.start(WireStart(SearchQuery("remote"), (SkillName("local"),), False))
        candidates = task.resolve_network(WireNetworkDecision(NetworkConsent.GRANTED))
        assert isinstance(candidates, PublicCandidates)
        required = task.select(
            remote_selection(str(candidates.response_token), "remote-1", include_local=True)
        )

        # When
        pending = task.activate(WireActivationDecision(ActivationConsent.NOT_REQUESTED))
        refused = task.activate(WireActivationDecision(ActivationConsent.REFUSED))

        # Then
        self.assertEqual(pending, required)
        self.assertEqual(
            refused,
            PublicExecutionReady((NativeLocalTarget(SkillName("local")),), ()),
        )
        self.assertEqual(provider.load_requests, ())

    def test_activation_grant_renders_remote_content(self) -> None:
        # Given
        remote = candidate_result("remote-1")
        content = RemoteSkillContent(
            remote.response_token,
            CandidateId("remote-1"),
            UntrustedText("remote body"),
        )
        provider = FakeProvider((remote,), (content,))
        task, _ = make_task(self.workspace, self.catalog, provider)
        task.start(WireStart(SearchQuery("remote"), (SkillName("local"),), False))
        candidates = task.resolve_network(WireNetworkDecision(NetworkConsent.GRANTED))
        assert isinstance(candidates, PublicCandidates)
        task.select(remote_selection(str(candidates.response_token), "remote-1"))

        # When
        result = task.activate(WireActivationDecision(ActivationConsent.GRANTED))

        # Then
        self.assertIsInstance(result, PublicExecutionReady)
        assert isinstance(result, PublicExecutionReady)
        self.assertEqual(
            result.targets,
            (
                RenderedRemoteTarget(
                    CandidateId("remote-1"),
                    remote.candidates[0].source_file,
                    "<<<REMOTE_SKILL_CONTENT>>>\nremote body\n<<<END_REMOTE_SKILL_CONTENT>>>\n",
                ),
            ),
        )
        self.assertEqual(len(provider.load_requests), 1)

    def test_remote_load_failure_and_delimiter_collision_preserve_local(self) -> None:
        cases = (
            (RemoteLoadUnavailable(UnavailableReason.TIMEOUT), UnavailableReason.TIMEOUT),
            ("collision", FacadeInvalidReason.DELIMITER_COLLISION),
        )
        for index, (load_result, reason) in enumerate(cases):
            with self.subTest(reason=reason):
                remote = candidate_result("remote-1")
                scripted = (
                    RemoteSkillContent(
                        remote.response_token,
                        CandidateId("remote-1"),
                        UntrustedText("<<<REMOTE_SKILL_CONTENT>>>"),
                    )
                    if load_result == "collision"
                    else load_result
                )
                provider = FakeProvider((remote,), (scripted,))
                task, _ = make_task(
                    self.workspace,
                    self.catalog,
                    provider,
                    handle=f"opaque-{index}",
                )
                task.start(WireStart(SearchQuery("remote"), (SkillName("local"),), False))
                candidates = task.resolve_network(WireNetworkDecision(NetworkConsent.GRANTED))
                assert isinstance(candidates, PublicCandidates)
                task.select(
                    remote_selection(str(candidates.response_token), "remote-1", include_local=True)
                )

                # When
                result = task.activate(WireActivationDecision(ActivationConsent.GRANTED))

                # Then
                self.assertIsInstance(result, PublicExecutionReady)
                assert isinstance(result, PublicExecutionReady)
                self.assertEqual(result.targets, (NativeLocalTarget(SkillName("local")),))
                self.assertEqual(result.degraded[0].reason, reason)


if __name__ == "__main__":
    unittest.main()
