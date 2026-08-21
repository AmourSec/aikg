from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from collections import deque
from io import StringIO
from pathlib import Path

from runtime.__main__ import serve
from runtime.contracts import (
    ActivationConsent,
    CandidateId,
    DisplayName,
    Endpoint,
    ExternalResponseToken,
    FacadeInvalidReason,
    HttpMethod,
    LocalCandidate,
    LocalScore,
    NativeLocalTarget,
    NetworkConsent,
    ProviderId,
    ProviderScore,
    RemoteCandidate,
    SearchQuery,
    SkillName,
    SourceFile,
    SourceRepo,
)
from runtime.coordinator import LocalExecutionMode
from runtime.rendering import RenderedRemoteTarget
from runtime.wire import (
    FacadeOutcome,
    PublicActivationRequired,
    PublicCandidates,
    PublicExecutionReady,
    PublicInvalid,
    SearchDisclosure,
    WireActivationDecision,
    WireNetworkDecision,
    WireSelection,
    WireStart,
)
from tests.facade_support import CatalogEntry, make_task, write_catalog
from tests.fakes import FakeProvider


class FakeTask:
    """Mutable call recorder for the host-adapter boundary."""

    def __init__(self, outcomes: tuple[FacadeOutcome, ...]) -> None:
        self._outcomes = deque(outcomes)
        self.calls: list[WireStart | WireNetworkDecision | WireSelection | WireActivationDecision] = []
        self.close_calls = 0

    def start(self, request: WireStart) -> FacadeOutcome:
        self.calls.append(request)
        return self._outcomes.popleft()

    def resolve_network(self, decision: WireNetworkDecision) -> FacadeOutcome:
        self.calls.append(decision)
        return self._outcomes.popleft()

    def select(self, request: WireSelection) -> FacadeOutcome:
        self.calls.append(request)
        return self._outcomes.popleft()

    def activate(self, decision: WireActivationDecision) -> FacadeOutcome:
        self.calls.append(decision)
        return self._outcomes.popleft()

    def close(self) -> None:
        self.close_calls += 1


class FakeFactory:
    def __init__(self, task: FakeTask) -> None:
        self.task = task
        self.calls = 0

    def __call__(self) -> FakeTask:
        self.calls += 1
        return self.task


def _run(lines: tuple[str, ...], outcomes: tuple[FacadeOutcome, ...]) -> tuple[list[str], FakeTask, int]:
    task = FakeTask(outcomes)
    stdout = StringIO()
    status = serve(StringIO("\n".join(lines) + "\n"), stdout, FakeFactory(task))
    return stdout.getvalue().splitlines(), task, status


class RuntimeEntrypointTests(unittest.TestCase):
    def test_valid_lifecycle_dispatches_typed_messages_and_stops_on_execution(self) -> None:
        # Given
        local = LocalCandidate(SkillName("local"), "Local skill", Path("/skill.md"), LocalScore(0.0))
        remote = RemoteCandidate(
            CandidateId("remote-1"), ProviderId("ascend-kg"), DisplayName("远程"),
            SourceRepo("org/repo"), SourceFile("skills/远程/SKILL.md"), ProviderScore("0.8")
        )
        outcomes: tuple[FacadeOutcome, ...] = (
            SearchDisclosure(Endpoint("https://ascend.wiki/search"), HttpMethod.POST, "{\"query\":\"查找\"}", (local,), (LocalExecutionMode.NATIVE,), ()),
            PublicCandidates((local,), (LocalExecutionMode.NATIVE,), (remote,), ExternalResponseToken("opaque"), ()),
            PublicActivationRequired((NativeLocalTarget(SkillName("local")),), (remote,), ()),
            PublicExecutionReady((RenderedRemoteTarget(CandidateId("remote-1"), remote.source_file, "远程正文\n原样保留"),), ()),
        )
        lines = (
            '{"type":"start","query":"查找","recalled_local_names":["local"],"local_only":false}',
            '{"type":"network_decision","consent":"granted"}',
            '{"type":"selection","local":[{"candidate_id":"local","execution_mode":"native"}],"remote":{"response_token":"opaque","provider_id":"ascend-kg","candidate_ids":["remote-1"]}}',
            '{"type":"activation_decision","consent":"granted"}',
            '{"type":"cancel"}',
        )

        # When
        encoded, task, status = _run(lines, outcomes)
        decoded = tuple(json.loads(line) for line in encoded)

        # Then
        self.assertEqual(status, 0)
        self.assertEqual(tuple(item["type"] for item in decoded), ("search_disclosure", "candidates", "activation_required", "execution_ready"))
        self.assertEqual(task.calls[0], WireStart(SearchQuery("查找"), (SkillName("local"),), False))
        self.assertEqual(task.calls[1], WireNetworkDecision(NetworkConsent.GRANTED))
        self.assertEqual(task.calls[3], WireActivationDecision(ActivationConsent.GRANTED))
        self.assertEqual(decoded[-1]["targets"][0]["rendered"], "远程正文\n原样保留")
        self.assertIn("远程正文", encoded[-1])
        self.assertEqual(task.close_calls, 1)

    def test_invalid_lines_emit_one_wire_invalid_without_dispatch(self) -> None:
        # Given
        lines = (
            "", "{", "[]", '{"type":"cancel","type":"start"}',
            '{"type":"unknown"}',
            '{"type":"start","query":"q","recalled_local_names":[]}',
            '{"type":"start","query":"q","recalled_local_names":[],"local_only":true,"extra":1}',
            '{"type":"start","query":"q","recalled_local_names":"local","local_only":true}',
            '{"type":"network_decision","consent":"later"}',
            '{"type":"selection","local":[{"candidate_id":"local","execution_mode":"native","extra":1}],"remote":null}',
        )

        # When
        encoded, task, status = _run(lines, ())
        decoded = tuple(json.loads(line) for line in encoded)

        # Then
        self.assertEqual(status, 0)
        self.assertEqual(len(decoded), len(lines))
        self.assertEqual(tuple(item["type"] for item in decoded), ("wire_invalid",) * len(lines))
        self.assertEqual(decoded[0]["reason"], "blank_line")
        self.assertEqual(decoded[3]["reason"], "duplicate_key")
        self.assertEqual(decoded[4]["reason"], "unknown_type")
        self.assertEqual(decoded[8]["reason"], "invalid_enum")
        self.assertEqual(task.calls, [])
        self.assertEqual(task.close_calls, 1)

    def test_valid_out_of_order_message_returns_facade_invalid(self) -> None:
        # Given
        line = '{"type":"selection","local":[],"remote":null}'

        # When
        encoded, task, _ = _run((line,), (PublicInvalid(FacadeInvalidReason.INVALID_TRANSITION),))

        # Then
        self.assertEqual(json.loads(encoded[0]), {"type": "invalid", "reason": "invalid_transition"})
        self.assertIsInstance(task.calls[0], WireSelection)

    def test_cancel_closes_and_stops_consuming(self) -> None:
        # Given / When
        encoded, task, status = _run((
            '{"type":"cancel"}',
            '{"type":"start","query":"ignored","recalled_local_names":[],"local_only":true}',
        ), ())

        # Then
        self.assertEqual(status, 0)
        self.assertEqual(encoded, ['{"type":"cancelled"}'])
        self.assertEqual(task.calls, [])
        self.assertEqual(task.close_calls, 1)

    def test_real_local_only_flow_reaches_coordinator_without_provider_search(self) -> None:
        # Given
        workspace = Path(self.enterContext(tempfile.TemporaryDirectory()))
        skill = workspace / "skills/local/SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("local", encoding="utf-8")
        catalog = write_catalog(workspace, (CatalogEntry("local", "skills/local/SKILL.md"),))
        provider = FakeProvider((), ())
        task, _ = make_task(workspace, catalog, provider)
        stdout = StringIO()
        stdin = StringIO(
            '{"type":"start","query":"local","recalled_local_names":["local"],"local_only":true}\n'
            '{"type":"selection","local":[{"candidate_id":"local","execution_mode":"native"}],"remote":null}\n'
        )

        # When
        status = serve(stdin, stdout, lambda: task)
        decoded = tuple(json.loads(line) for line in stdout.getvalue().splitlines())

        # Then
        self.assertEqual(status, 0)
        self.assertEqual(tuple(item["type"] for item in decoded), ("candidates", "execution_ready"))
        self.assertEqual(decoded[-1]["targets"], [{"type": "native_local", "skill_name": "local"}])
        self.assertEqual(provider.search_requests, ())

    def test_module_help_succeeds_without_required_arguments(self) -> None:
        # Given / When
        result = subprocess.run(
            (sys.executable, "-m", "runtime", "--help"),
            cwd=Path(__file__).parents[1],
            check=False,
            capture_output=True,
            text=True,
        )

        # Then
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--catalog", result.stdout)
        self.assertIn("--workspace-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
