from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

from .contracts import (
    CandidateId,
    FacadeInvalidReason,
    RemoteCandidate,
    RemoteSkillContent,
    SourceFile,
)

_START_DELIMITER: Final = "<<<REMOTE_SKILL_CONTENT>>>"
_END_DELIMITER: Final = "<<<END_REMOTE_SKILL_CONTENT>>>"


@dataclass(frozen=True, slots=True)
class RenderedRemoteTarget:
    candidate_id: CandidateId
    source_file: SourceFile
    rendered: str


@dataclass(frozen=True, slots=True)
class RemoteRenderInvalid:
    candidate_id: CandidateId
    reason: FacadeInvalidReason


RemoteRenderResult: TypeAlias = RenderedRemoteTarget | RemoteRenderInvalid


def render_remote_target(
    candidate: RemoteCandidate,
    content: RemoteSkillContent,
) -> RemoteRenderResult:
    text = content.content
    if _START_DELIMITER in text or _END_DELIMITER in text:
        return RemoteRenderInvalid(
            candidate_id=candidate.candidate_id,
            reason=FacadeInvalidReason.DELIMITER_COLLISION,
        )
    separator = "" if text.endswith("\n") else "\n"
    rendered = f"{_START_DELIMITER}\n{text}{separator}{_END_DELIMITER}\n"
    return RenderedRemoteTarget(
        candidate_id=candidate.candidate_id,
        source_file=candidate.source_file,
        rendered=rendered,
    )
