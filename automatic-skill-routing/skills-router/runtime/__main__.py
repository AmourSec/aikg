from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO, assert_never

from runtime.contracts import SkillName
from runtime.facade import RouterTask, TaskRouterConfig
from runtime.ndjson import (
    DispatchMessage,
    WireCancel,
    WireInvalid,
    parse_line,
)
from runtime.ndjson_output import Cancelled, HostOutcome, encode_outcome
from runtime.wire import (
    FacadeOutcome,
    PublicActivationRequired,
    PublicCandidates,
    PublicDegraded,
    PublicExecutionReady,
    PublicInvalid,
    SearchDisclosure,
    WireActivationDecision,
    WireNetworkDecision,
    WireSelection,
    WireStart,
)

__all__ = ("RouterTaskFactory", "main", "serve")


class RouterTaskHandle(Protocol):
    def start(self, request: WireStart) -> FacadeOutcome: ...

    def resolve_network(self, decision: WireNetworkDecision) -> FacadeOutcome: ...

    def select(self, request: WireSelection) -> FacadeOutcome: ...

    def activate(self, decision: WireActivationDecision) -> FacadeOutcome: ...

    def close(self) -> None: ...


class RouterTaskFactory(Protocol):
    def __call__(self) -> RouterTaskHandle: ...


@dataclass(frozen=True, slots=True)
class _EnvironmentTaskFactory:
    config: TaskRouterConfig

    def __call__(self) -> RouterTask:
        return RouterTask.from_environment(self.config)


def serve(stdin: TextIO, stdout: TextIO, factory: RouterTaskFactory) -> int:
    task = factory()
    closed = False
    try:
        for line in stdin:
            message = parse_line(line)
            match message:
                case WireInvalid():
                    outcome: HostOutcome = message
                    terminal = False
                case WireCancel():
                    task.close()
                    closed = True
                    outcome = Cancelled()
                    terminal = True
                case WireStart() | WireNetworkDecision() | WireSelection() | WireActivationDecision():
                    facade_outcome = _dispatch(task, message)
                    outcome = facade_outcome
                    terminal = _is_terminal(facade_outcome)
                case unreachable:
                    assert_never(unreachable)

            stdout.write(encode_outcome(outcome) + "\n")
            stdout.flush()
            if terminal:
                if not closed:
                    task.close()
                    closed = True
                return 0
        return 0
    finally:
        if not closed:
            task.close()


def _dispatch(task: RouterTaskHandle, message: DispatchMessage) -> FacadeOutcome:
    match message:
        case WireStart():
            return task.start(message)
        case WireNetworkDecision():
            return task.resolve_network(message)
        case WireSelection():
            return task.select(message)
        case WireActivationDecision():
            return task.activate(message)
        case unreachable:
            assert_never(unreachable)


def _is_terminal(outcome: FacadeOutcome) -> bool:
    match outcome:
        case PublicExecutionReady():
            return True
        case SearchDisclosure() | PublicCandidates() | PublicActivationRequired() | PublicDegraded() | PublicInvalid():
            return False
        case unreachable:
            assert_never(unreachable)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m runtime",
        description="Serve one skills-router task over NDJSON.",
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--native-skill", action="append", default=[])
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = TaskRouterConfig(
        catalog_path=Path(args.catalog),
        workspace_root=Path(args.workspace_root),
        native_skill_names=frozenset(
            SkillName(name) for name in args.native_skill
        ),
    )
    return serve(sys.stdin, sys.stdout, _EnvironmentTaskFactory(config))


if __name__ == "__main__":
    raise SystemExit(main())
