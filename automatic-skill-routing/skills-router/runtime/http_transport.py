from __future__ import annotations

import socket
from contextlib import closing
from email.message import Message
from http.client import HTTPException
from typing import BinaryIO, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, OpenerDirector, Request, build_opener

from runtime.contracts import (
    HeaderName,
    HeaderValue,
    Headers,
    TransportRequest,
    TransportResponse,
    TransportResult,
    TransportTimeout,
    TransportUnavailable,
    UnavailableReason,
)


class _ResponseBody(Protocol):
    headers: Message

    def read(self, limit: int) -> bytes: ...

    def close(self) -> None: ...


class _HttpResponse(_ResponseBody, Protocol):
    status: int


class _Opener(Protocol):
    def open(self, request: Request, timeout: float) -> _HttpResponse: ...


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> None:
        return None


class UrllibTransport:
    def __init__(self, opener: _Opener | None = None) -> None:
        self._opener = opener if opener is not None else build_opener(_RejectRedirects())

    def send(self, request: TransportRequest) -> TransportResult:
        urllib_request = Request(
            url=str(request.endpoint),
            data=request.body or None,
            headers={str(name): str(value) for name, value in request.headers},
            method=request.method.value,
        )
        try:
            response = self._opener.open(
                urllib_request,
                timeout=float(request.timeout_seconds),
            )
        except HTTPError as error:
            return _read_response(error, error.code, request)
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                return TransportTimeout(request.timeout_seconds)
            return TransportUnavailable(UnavailableReason.SERVICE)
        except (TimeoutError, socket.timeout):
            return TransportTimeout(request.timeout_seconds)
        except (OSError, HTTPException):
            return TransportUnavailable(UnavailableReason.SERVICE)
        return _read_response(response, response.status, request)


def _read_response(
    response: _ResponseBody,
    status_code: int,
    request: TransportRequest,
) -> TransportResult:
    try:
        with closing(response):
            return TransportResponse(
                status_code=status_code,
                headers=_headers(response.headers),
                body=response.read(int(request.max_response_bytes) + 1),
            )
    except (TimeoutError, socket.timeout):
        return TransportTimeout(request.timeout_seconds)
    except (OSError, HTTPException):
        return TransportUnavailable(UnavailableReason.SERVICE)


def _headers(message: Message) -> Headers:
    return tuple(
        (HeaderName(name), HeaderValue(value))
        for name, value in message.items()
    )
