from __future__ import annotations

import socket
import unittest
from email.message import Message
from http.client import HTTPException
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from runtime.contracts import (
    ByteLimit,
    Endpoint,
    HeaderName,
    HeaderValue,
    HttpMethod,
    Seconds,
    TransportRequest,
    TransportResponse,
    TransportTimeout,
    TransportUnavailable,
    UnavailableReason,
)
from runtime.http_transport import UrllibTransport


class FakeHttpResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"
        self._body = BytesIO(body)
        self.read_limits: list[int] = []

    def read(self, limit: int) -> bytes:
        self.read_limits.append(limit)
        return self._body.read(limit)

    def close(self) -> None:
        self._body.close()


class FailingHttpResponse(FakeHttpResponse):
    def __init__(self, failure: OSError | HTTPException) -> None:
        super().__init__(200, b"")
        self._failure = failure

    def read(self, limit: int) -> bytes:
        self.read_limits.append(limit)
        raise self._failure


class FakeOpener:
    def __init__(self, result: FakeHttpResponse | HTTPError | URLError) -> None:
        self.result = result
        self.requests = []

    def open(self, request, timeout: float):
        self.requests.append((request, timeout))
        if isinstance(self.result, FakeHttpResponse):
            return self.result
        raise self.result


def transport_request(limit: int = 8) -> TransportRequest:
    return TransportRequest(
        method=HttpMethod.POST,
        endpoint=Endpoint("https://ascend.wiki/search"),
        headers=((HeaderName("X-API-Key"), HeaderValue("secret")),),
        body=b"{}",
        timeout_seconds=Seconds(10.0),
        max_response_bytes=ByteLimit(limit),
    )


class UrllibTransportTests(unittest.TestCase):
    def test_send_reads_at_most_limit_plus_one_and_preserves_request(self) -> None:
        # Given
        response = FakeHttpResponse(200, b"1234567890")
        opener = FakeOpener(response)
        transport = UrllibTransport(opener=opener)

        # When
        result = transport.send(transport_request())

        # Then
        self.assertEqual(
            result,
            TransportResponse(
                status_code=200,
                headers=((HeaderName("Content-Type"), HeaderValue("application/json")),),
                body=b"123456789",
            ),
        )
        self.assertEqual(response.read_limits, [9])
        sent, timeout = opener.requests[0]
        self.assertEqual(sent.full_url, "https://ascend.wiki/search")
        self.assertEqual(sent.get_method(), "POST")
        self.assertEqual(sent.data, b"{}")
        self.assertEqual(sent.get_header("X-api-key"), "secret")
        self.assertEqual(timeout, 10.0)

    def test_send_returns_http_errors_as_bounded_responses(self) -> None:
        # Given
        error = HTTPError(
            "https://ascend.wiki/search",
            429,
            "rate limited",
            Message(),
            BytesIO(b"1234567890"),
        )
        transport = UrllibTransport(opener=FakeOpener(error))

        # When
        result = transport.send(transport_request())

        # Then
        self.assertEqual(
            result,
            TransportResponse(status_code=429, headers=(), body=b"123456789"),
        )

    def test_send_maps_timeout_and_connection_failure(self) -> None:
        cases = (
            (
                URLError(socket.timeout("slow")),
                TransportTimeout(timeout_seconds=Seconds(10.0)),
            ),
            (
                URLError(ConnectionRefusedError("down")),
                TransportUnavailable(reason=UnavailableReason.SERVICE),
            ),
        )
        for error, expected in cases:
            with self.subTest(error=error):
                # Given
                transport = UrllibTransport(opener=FakeOpener(error))

                # When
                result = transport.send(transport_request())

                # Then
                self.assertEqual(result, expected)

    def test_send_maps_failures_while_reading_successful_response(self) -> None:
        cases = (
            (socket.timeout("slow"), TransportTimeout(Seconds(10.0))),
            (
                ConnectionResetError("reset"),
                TransportUnavailable(UnavailableReason.SERVICE),
            ),
            (
                HTTPException("broken response"),
                TransportUnavailable(UnavailableReason.SERVICE),
            ),
        )
        for failure, expected in cases:
            with self.subTest(failure=failure):
                # Given
                response = FailingHttpResponse(failure)
                transport = UrllibTransport(opener=FakeOpener(response))

                # When
                result = transport.send(transport_request())

                # Then
                self.assertEqual(result, expected)

    def test_send_maps_failures_while_reading_http_error_body(self) -> None:
        cases = (
            (socket.timeout("slow"), TransportTimeout(Seconds(10.0))),
            (OSError("read failed"), TransportUnavailable(UnavailableReason.SERVICE)),
            (
                HTTPException("broken response"),
                TransportUnavailable(UnavailableReason.SERVICE),
            ),
        )
        for failure, expected in cases:
            with self.subTest(failure=failure):
                # Given
                error = HTTPError(
                    "https://ascend.wiki/search",
                    503,
                    "service unavailable",
                    Message(),
                    FailingHttpResponse(failure),
                )
                transport = UrllibTransport(opener=FakeOpener(error))

                # When
                result = transport.send(transport_request())

                # Then
                self.assertEqual(result, expected)

    def test_default_transport_rejects_redirects(self) -> None:
        # Given
        opener = FakeOpener(FakeHttpResponse(200, b"ok"))

        # When
        with patch("runtime.http_transport.build_opener", return_value=opener) as build:
            UrllibTransport()

        # Then
        handler = build.call_args.args[0]
        self.assertIsNone(
            handler.redirect_request(
                Request("https://ascend.wiki/search"),
                BytesIO(),
                302,
                "redirect",
                Message(),
                "https://evil.invalid",
            ),
        )


if __name__ == "__main__":
    unittest.main()
