from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from zoneinfo import ZoneInfo

from scripts import update_pageviews


class UpdatePageviewsTests(unittest.TestCase):
    def test_request_json_exposes_goatcounter_http_error_details(self) -> None:
        # Given GoatCounter returns a structured HTTP error
        response = HTTPError(
            "https://amoursec.goatcounter.com/api/v0/stats/hits",
            404,
            "Not Found",
            {},
            BytesIO(b'{"error":"site not found"}'),
        )

        # When the API request is made
        # Then the failure retains the status and safe response detail
        with (
            patch.object(update_pageviews, "urlopen", side_effect=response),
            self.assertRaises(update_pageviews.GoatCounterHttpError) as caught,
        ):
            update_pageviews.request_json(
                "https://amoursec.goatcounter.com/api/v0/stats/hits",
                {},
                "secret",
            )

        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.detail, "site not found")

    def test_normalize_path_merges_clean_urls(self) -> None:
        # Given equivalent clean and decorated article paths
        values = ("", "/docs", "/docs/?q=1#x")

        # When the paths are normalized
        normalized = tuple(update_pageviews.normalize_path(value) for value in values)

        # Then equivalent paths use the canonical trailing-slash form
        self.assertEqual(normalized, ("/", "/docs/", "/docs/"))

    def test_fetch_pageviews_paginates_and_merges_paths(self) -> None:
        # Given two GoatCounter pages containing two spellings of one article path
        calls: list[tuple[str, dict[str, update_pageviews.QueryValue]]] = []

        def fake_request(
            url: str,
            params: dict[str, update_pageviews.QueryValue],
            api_key: str,
        ) -> update_pageviews.JsonValue:
            del api_key
            calls.append((url, params))
            if "exclude_paths" not in params:
                return {
                    "hits": [
                        {
                            "count": 2,
                            "path_id": 11,
                            "path": "/article",
                            "event": False,
                            "title": "Article",
                            "max": 2,
                            "stats": [],
                        },
                        {
                            "count": 99,
                            "path_id": 12,
                            "path": "download",
                            "event": True,
                            "title": "Download",
                            "max": 99,
                            "stats": [],
                        },
                    ],
                    "total": 101,
                    "more": True,
                }
            return {
                "hits": [
                    {
                        "count": 3,
                        "path_id": 13,
                        "path": "/article/",
                        "event": False,
                        "title": "Article",
                        "max": 3,
                        "stats": [],
                    }
                ],
                "total": 3,
                "more": False,
            }

        # When all pages are fetched
        pages = update_pageviews.fetch_pageviews(
            "secret",
            "amoursec",
            request_json=fake_request,
            page_size=100,
            now=datetime(2026, 9, 1, 7, 42, tzinfo=UTC),
        )

        # Then aliases are merged, events are ignored, and returned IDs paginate
        self.assertEqual(pages, {"/article/": 5})
        self.assertEqual(
            calls,
            [
                (
                    "https://amoursec.goatcounter.com/api/v0/stats/hits",
                    {
                        "start": "1970-01-01T00:00:00Z",
                        "end": "2026-09-01T07:00:00Z",
                        "limit": 100,
                    },
                ),
                (
                    "https://amoursec.goatcounter.com/api/v0/stats/hits",
                    {
                        "start": "1970-01-01T00:00:00Z",
                        "end": "2026-09-01T07:00:00Z",
                        "limit": 100,
                        "exclude_paths": (11, 12),
                    },
                ),
            ],
        )

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
        # Given an existing snapshot and an invalid GoatCounter hit
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "pageviews.json"
            target.write_text(
                '{"schema_version":1,"updated_at":null,"pages":{}}\n',
                encoding="utf-8",
            )

            def fake_request(
                url: str,
                params: dict[str, update_pageviews.QueryValue],
                api_key: str,
            ) -> update_pageviews.JsonValue:
                del url, params, api_key
                return {
                    "hits": [
                        {
                            "count": "five",
                            "path_id": 11,
                            "path": "/article/",
                            "event": False,
                            "title": "Article",
                            "max": 5,
                            "stats": [],
                        }
                    ],
                    "total": 5,
                    "more": False,
                }

            before = target.read_bytes()

            # When a snapshot sync receives the invalid response
            # Then the sync fails without replacing the existing snapshot
            with self.assertRaises(update_pageviews.GoatCounterDataError):
                update_pageviews.sync_snapshot(
                    target,
                    "secret",
                    "amoursec",
                    request_json=fake_request,
                )

            self.assertEqual(target.read_bytes(), before)

    def test_pagination_rejects_more_without_new_path_ids(self) -> None:
        # Given a GoatCounter response claiming another page without any hits
        def fake_request(
            url: str,
            params: dict[str, update_pageviews.QueryValue],
            api_key: str,
        ) -> update_pageviews.JsonValue:
            del url, params, api_key
            return {"hits": [], "total": 0, "more": True}

        # When the inconsistent response is fetched
        # Then fetching stops instead of looping forever
        with self.assertRaises(update_pageviews.GoatCounterDataError):
            update_pageviews.fetch_pageviews(
                "secret",
                "amoursec",
                request_json=fake_request,
            )

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
