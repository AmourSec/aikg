from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

from scripts import update_pageviews

ENDPOINT = "https://amoursec.goatcounter.com/api/v0/stats/hits"


class UpdatePageviewsRetryTests(unittest.TestCase):
    def test_request_json_retries_transient_404_then_succeeds(self) -> None:
        # Given GoatCounter returns one transient 404 before a valid response
        transient_error = HTTPError(
            ENDPOINT,
            404,
            "Not Found",
            {},
            BytesIO(b"<html><title>GoatCounter Error!</title></html>"),
        )
        response = BytesIO(b'{"hits":[],"total":0,"more":false}')

        # When the API request is made
        with (
            patch.object(
                update_pageviews,
                "urlopen",
                side_effect=(transient_error, response),
            ) as opener,
            patch("time.sleep", return_value=None),
        ):
            try:
                result = update_pageviews.request_json(ENDPOINT, {}, "secret")
            except update_pageviews.GoatCounterHttpError:
                result = None

        # Then the retry returns the successful response
        self.assertEqual(opener.call_count, 2)
        self.assertEqual(result, {"hits": [], "total": 0, "more": False})

    def test_request_json_stops_after_three_transient_failures(self) -> None:
        # Given GoatCounter keeps returning a retryable server error
        responses = tuple(
            HTTPError(
                ENDPOINT,
                503,
                "Service Unavailable",
                {},
                BytesIO(b'{"error":"temporarily unavailable"}'),
            )
            for _ in range(3)
        )

        # When the API request exhausts its retry budget
        # Then the final failure remains visible after exactly three attempts
        with (
            patch.object(
                update_pageviews,
                "urlopen",
                side_effect=responses,
            ) as opener,
            patch("time.sleep", return_value=None),
            self.assertRaises(update_pageviews.GoatCounterHttpError) as caught,
        ):
            update_pageviews.request_json(ENDPOINT, {}, "secret")

        self.assertEqual(opener.call_count, 3)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.detail, "temporarily unavailable")


if __name__ == "__main__":
    unittest.main()
