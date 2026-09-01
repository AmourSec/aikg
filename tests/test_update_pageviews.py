from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts import update_pageviews


class UpdatePageviewsTests(unittest.TestCase):
    def test_normalize_path_merges_clean_urls(self) -> None:
        # Given equivalent clean and decorated article paths
        values = ("", "/docs", "/docs/?q=1#x")

        # When the paths are normalized
        normalized = tuple(update_pageviews.normalize_path(value) for value in values)

        # Then equivalent paths use the canonical trailing-slash form
        self.assertEqual(normalized, ("/", "/docs/", "/docs/"))

    def test_fetch_pageviews_paginates_and_merges_paths(self) -> None:
        # Given two Umami pages containing two spellings of one article path
        calls: list[tuple[str, dict[str, str | int]]] = []

        def fake_request(
            path: str,
            params: dict[str, str | int],
            api_key: str,
        ) -> update_pageviews.JsonValue:
            del api_key
            calls.append((path, params))
            if path.endswith("/daterange"):
                return {
                    "startDate": "2026-08-01T00:00:00Z",
                    "endDate": "2026-09-01T00:00:00Z",
                }
            if params["offset"] == 0:
                return [
                    {"name": "/article", "pageviews": 2},
                    {"name": "/article/", "pageviews": 3},
                ]
            return []

        # When all pages are fetched
        pages = update_pageviews.fetch_pageviews(
            "secret",
            "e6bcb0cd-aee7-4383-8557-9cf7564c86a0",
            request_json=fake_request,
            page_size=2,
        )

        # Then aliases are merged and pagination advances by the requested size
        self.assertEqual(pages, {"/article/": 5})
        self.assertEqual([call[1]["offset"] for call in calls[1:]], [0, 2])

    def test_write_snapshot_keeps_timestamp_when_counts_do_not_change(self) -> None:
        # Given an existing snapshot with unchanged counts
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "pageviews.json"
            target.write_text(
                '{"schema_version":1,"updated_at":"2026-09-01T23:17:00+08:00",'
                '"pages":{"/article/":5}}\n',
                encoding="utf-8",
            )

            # When the same counts are written a day later
            changed = update_pageviews.write_snapshot(
                target,
                {"/article/": 5},
                now=datetime(2026, 9, 2, 23, 17, tzinfo=ZoneInfo("Asia/Taipei")),
            )

            # Then no file change is reported and the original timestamp remains
            self.assertFalse(changed)
            self.assertIn("2026-09-01T23:17:00+08:00", target.read_text())

    def test_invalid_pageview_does_not_replace_snapshot(self) -> None:
        # Given an existing snapshot and an invalid Umami metric
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "pageviews.json"
            target.write_text(
                '{"schema_version":1,"updated_at":null,"pages":{}}\n',
                encoding="utf-8",
            )

            def fake_request(
                path: str,
                params: dict[str, str | int],
                api_key: str,
            ) -> update_pageviews.JsonValue:
                del params, api_key
                if path.endswith("/daterange"):
                    return {
                        "startDate": "2026-08-01T00:00:00Z",
                        "endDate": "2026-09-01T00:00:00Z",
                    }
                return [{"name": "/article/", "pageviews": "five"}]

            before = target.read_bytes()

            # When the invalid response is fetched
            with self.assertRaises(update_pageviews.UmamiDataError):
                update_pageviews.fetch_pageviews(
                    "secret",
                    "e6bcb0cd-aee7-4383-8557-9cf7564c86a0",
                    request_json=fake_request,
                )

            # Then the existing snapshot remains byte-for-byte unchanged
            self.assertEqual(target.read_bytes(), before)

    def test_write_snapshot_uses_stable_schema_and_sorted_paths(self) -> None:
        # Given counts arriving in a non-deterministic order
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "pageviews.json"

            # When a new snapshot is written
            changed = update_pageviews.write_snapshot(
                target,
                {"/z/": 1, "/a/": 2},
                now=datetime(2026, 9, 1, 23, 17, tzinfo=ZoneInfo("Asia/Taipei")),
            )

            # Then the schema is complete and paths are written deterministically
            self.assertTrue(changed)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["updated_at"], "2026-09-01T23:17:00+08:00")
            self.assertEqual(list(payload["pages"]), ["/a/", "/z/"])


if __name__ == "__main__":
    unittest.main()
