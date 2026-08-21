from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from runtime.ascend_kg import AscendKgProvider
from runtime.contracts import LocalScore, SkillName
from runtime.coordinator import LocalExecutionMode
from runtime.http_transport import UrllibTransport
from runtime.local_catalog import (
    LocalCatalogDegradation,
    LocalCatalogDegradationReason,
    LocalCatalogInvalidRequestError,
    LocalCatalogInvalidRequestReason,
    LocalCatalogParseError,
    LocalCatalogParseReason,
    LocalCatalogRequest,
    load_local_candidates,
)


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    name: str
    path: str
    description: str = "Local catalog test skill."


def _write_catalog(workspace: Path, entries: tuple[CatalogEntry, ...]) -> Path:
    catalog_path = workspace / "catalog.json"
    catalog_path.write_text(
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
    return catalog_path


def _request(
    recalled_ids: tuple[str, ...],
    native_skill_names: tuple[str, ...] = (),
) -> LocalCatalogRequest:
    return LocalCatalogRequest(
        recalled_ids=tuple(SkillName(name) for name in recalled_ids),
        native_skill_names=frozenset(SkillName(name) for name in native_skill_names),
    )


class LocalCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        base = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.workspace = base / "workspace"
        self.workspace.mkdir()
        self.outside = base / "outside"
        self.outside.mkdir()

    def test_load_preserves_recalled_order_and_native_wins_over_missing_cache(self) -> None:
        # Given
        source = self.workspace / "skills/path-only/SKILL.md"
        source.parent.mkdir(parents=True)
        source.write_text("path skill", encoding="utf-8")
        catalog = _write_catalog(
            self.workspace,
            (
                CatalogEntry("native", ".skills-cache/missing/SKILL.md"),
                CatalogEntry("path-only", "skills/path-only/SKILL.md"),
                CatalogEntry("cache-missing", ".skills-cache/absent/SKILL.md"),
            ),
        )

        # When
        result = load_local_candidates(
            _request(
                ("path-only", "native", "cache-missing"),
                native_skill_names=("native",),
            ),
            catalog,
            self.workspace,
        )

        # Then
        self.assertEqual(
            tuple(candidate.name for candidate in result.candidates),
            (SkillName("path-only"), SkillName("native")),
        )
        self.assertEqual(
            result.modes,
            (LocalExecutionMode.PATH, LocalExecutionMode.NATIVE),
        )
        self.assertEqual(result.candidates[0].path, source.resolve())
        self.assertEqual(result.candidates[0].score, LocalScore(0.0))
        self.assertEqual(
            result.degraded,
            (
                LocalCatalogDegradation(
                    candidate_id=SkillName("cache-missing"),
                    path=(self.workspace / ".skills-cache/absent/SKILL.md").resolve(),
                    reason=LocalCatalogDegradationReason.MISSING,
                ),
            ),
        )
        self.assertTrue(result.absent_cache)

    def test_unsafe_and_unavailable_paths_degrade_without_candidates(self) -> None:
        # Given
        outside_file = self.outside / "SKILL.md"
        outside_file.write_text("outside", encoding="utf-8")
        directory = self.workspace / "skills/directory"
        directory.mkdir(parents=True)
        cases = (
            (
                "traversal",
                "../outside/SKILL.md",
                LocalCatalogDegradationReason.TRAVERSAL,
            ),
            (
                "absolute",
                str(outside_file),
                LocalCatalogDegradationReason.ABSOLUTE_ESCAPE,
            ),
            (
                "directory",
                "skills/directory",
                LocalCatalogDegradationReason.DIRECTORY,
            ),
        )

        for name, path, reason in cases:
            with self.subTest(name=name):
                catalog = _write_catalog(
                    self.workspace,
                    (CatalogEntry(name, path),),
                )

                # When
                result = load_local_candidates(
                    _request((name,)),
                    catalog,
                    self.workspace,
                )

                # Then
                self.assertEqual(result.candidates, ())
                self.assertEqual(result.degraded[0].reason, reason)

    def test_symlink_escape_degrades_where_symlinks_are_supported(self) -> None:
        # Given
        outside_file = self.outside / "SKILL.md"
        outside_file.write_text("outside", encoding="utf-8")
        link = self.workspace / "skills/link.md"
        link.parent.mkdir(parents=True)
        try:
            link.symlink_to(outside_file)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"symlinks are unavailable: {error}")
        catalog = _write_catalog(
            self.workspace,
            (CatalogEntry("linked", "skills/link.md"),),
        )

        # When
        result = load_local_candidates(
            _request(("linked",)),
            catalog,
            self.workspace,
        )

        # Then
        self.assertEqual(result.candidates, ())
        self.assertEqual(
            result.degraded[0].reason,
            LocalCatalogDegradationReason.SYMLINK_ESCAPE,
        )

    def test_unknown_and_duplicate_recalled_ids_are_typed_invalid_requests(self) -> None:
        # Given
        catalog = _write_catalog(
            self.workspace,
            (CatalogEntry("known", "skills/known/SKILL.md"),),
        )
        cases = (
            (
                _request(("unknown",)),
                LocalCatalogInvalidRequestReason.UNKNOWN_RECALLED_ID,
                SkillName("unknown"),
            ),
            (
                _request(("known", "known")),
                LocalCatalogInvalidRequestReason.DUPLICATE_RECALLED_ID,
                SkillName("known"),
            ),
        )

        for request, reason, candidate_id in cases:
            with self.subTest(reason=reason):
                # When / Then
                with self.assertRaises(LocalCatalogInvalidRequestError) as raised:
                    load_local_candidates(request, catalog, self.workspace)
                self.assertEqual(raised.exception.reason, reason)
                self.assertEqual(raised.exception.candidate_id, candidate_id)

    def test_malformed_catalog_boundary_raises_typed_parse_error(self) -> None:
        # Given
        catalog = self.workspace / "catalog.json"
        cases = (
            ("not-json", LocalCatalogParseReason.INVALID_JSON),
            (json.dumps([]), LocalCatalogParseReason.INVALID_SCHEMA),
            (json.dumps({"skills": {}}), LocalCatalogParseReason.INVALID_SCHEMA),
            (
                json.dumps(
                    {"skills": [{"name": "broken", "description": "text"}]}
                ),
                LocalCatalogParseReason.INVALID_SCHEMA,
            ),
        )

        for content, reason in cases:
            with self.subTest(reason=reason):
                catalog.write_text(content, encoding="utf-8")

                # When / Then
                with self.assertRaises(LocalCatalogParseError) as raised:
                    load_local_candidates(_request(("broken",)), catalog, self.workspace)
                self.assertEqual(raised.exception.reason, reason)
                self.assertEqual(raised.exception.path, catalog)

    def test_all_missing_candidates_return_empty_ordered_fallback(self) -> None:
        # Given
        catalog = _write_catalog(
            self.workspace,
            (
                CatalogEntry("local-missing", "skills/missing/SKILL.md"),
                CatalogEntry("cache-missing", ".skills-cache/missing/SKILL.md"),
            ),
        )

        # When
        result = load_local_candidates(
            _request(("cache-missing", "local-missing")),
            catalog,
            self.workspace,
        )

        # Then
        self.assertEqual((result.candidates, result.modes), ((), ()))
        self.assertEqual(
            tuple(item.candidate_id for item in result.degraded),
            (SkillName("cache-missing"), SkillName("local-missing")),
        )
        self.assertTrue(result.absent_cache)

    def test_loading_is_read_only_and_does_not_involve_remote_runtime(self) -> None:
        # Given
        source = self.workspace / "skills/local/SKILL.md"
        source.parent.mkdir(parents=True)
        source.write_text("local", encoding="utf-8")
        catalog = _write_catalog(
            self.workspace,
            (CatalogEntry("local", "skills/local/SKILL.md"),),
        )
        before_catalog = (catalog.read_bytes(), catalog.stat().st_mtime_ns)
        before_source = (source.read_bytes(), source.stat().st_mtime_ns)

        # When
        with (
            patch.object(AscendKgProvider, "search", side_effect=AssertionError),
            patch.object(AscendKgProvider, "load_skill", side_effect=AssertionError),
            patch.object(UrllibTransport, "send", side_effect=AssertionError),
        ):
            result = load_local_candidates(
                _request(("local",)),
                catalog,
                self.workspace,
            )

        # Then
        self.assertEqual(result.modes, (LocalExecutionMode.PATH,))
        self.assertEqual(
            (before_catalog, before_source),
            (
                (catalog.read_bytes(), catalog.stat().st_mtime_ns),
                (source.read_bytes(), source.stat().st_mtime_ns),
            ),
        )
