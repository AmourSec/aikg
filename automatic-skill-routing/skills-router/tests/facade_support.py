from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from runtime.contracts import (
    CandidateId,
    Candidates,
    DisplayName,
    ExternalResponseToken,
    ProviderId,
    RemoteCandidate,
    ResponseToken,
    SkillName,
    SourceFile,
    SourceRepo,
)
from runtime.coordinator import Coordinator, LocalExecutionMode
from runtime.facade import RouterTask, TaskRouterConfig
from runtime.token_registry import ResponseTokenRegistry
from runtime.wire import WireLocalSelection, WireRemoteSelection, WireSelection
from tests.fakes import FakeProvider


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    name: str
    path: str
    description: str = "Facade test skill."


class DeterministicIssuer:
    def __init__(self, *handles: str) -> None:
        self._handles = iter(handles)

    def __call__(self) -> str:
        return next(self._handles)


def write_catalog(workspace: Path, entries: tuple[CatalogEntry, ...]) -> Path:
    path = workspace / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "skills": [
                    {
                        "name": entry.name,
                        "description": entry.description,
                        "path": entry.path,
                    }
                    for entry in entries
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def remote_candidate(candidate_id: str = "remote-1") -> RemoteCandidate:
    return RemoteCandidate(
        candidate_id=CandidateId(candidate_id),
        provider_id=ProviderId("ascend-kg"),
        display_name=DisplayName(f"Remote {candidate_id}"),
        source_repo=SourceRepo("org/skills"),
        source_file=SourceFile(f"skills/{candidate_id}/SKILL.md"),
    )


def candidate_result(*candidate_ids: str) -> Candidates:
    ids = tuple(CandidateId(value) for value in candidate_ids)
    return Candidates(
        response_token=ResponseToken(ids),
        candidates=tuple(remote_candidate(value) for value in candidate_ids),
    )


def make_task(
    workspace: Path,
    catalog_path: Path,
    provider: FakeProvider,
    *,
    native_names: tuple[str, ...] = ("local",),
    handle: str = "opaque-task-handle",
) -> tuple[RouterTask, ResponseTokenRegistry]:
    registry = ResponseTokenRegistry(DeterministicIssuer(handle))
    task = RouterTask(
        TaskRouterConfig(
            catalog_path=catalog_path,
            workspace_root=workspace,
            native_skill_names=frozenset(
                SkillName(name) for name in native_names
            ),
        ),
        Coordinator(provider),
        registry,
    )
    return task, registry


def local_selection(
    name: str = "local",
    mode: LocalExecutionMode = LocalExecutionMode.NATIVE,
) -> WireSelection:
    return WireSelection(
        local=(WireLocalSelection(SkillName(name), mode),),
        remote=None,
    )


def remote_selection(
    handle: str,
    *candidate_ids: str,
    include_local: bool = False,
) -> WireSelection:
    local = (
        (WireLocalSelection(SkillName("local"), LocalExecutionMode.NATIVE),)
        if include_local
        else ()
    )
    return WireSelection(
        local=local,
        remote=WireRemoteSelection(
            response_token=ExternalResponseToken(handle),
            provider_id=ProviderId("ascend-kg"),
            candidate_ids=tuple(CandidateId(value) for value in candidate_ids),
        ),
    )
