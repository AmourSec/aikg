from __future__ import annotations

import os
import time
from typing import Final, assert_never
from urllib.parse import quote

from runtime.ascend_kg_parsing import parse_search_response, parse_skill_response
from runtime.contracts import (
    ActivationConsent, Ambiguous, ByteLimit, Candidates, Endpoint, HeaderName,
    HeaderValue, HttpMethod, InvalidResponse, InvalidResponseReason, LoadRequest,
    NetworkConsent, NoMatch, RemoteLoadInvalid, RemoteLoadResult,
    RemoteLoadUnavailable, ResponseToken, SearchRequest, SearchResult, Seconds,
    Sleeper, Transport, TransportRequest, TransportResponse, TransportResult,
    TransportTimeout, TransportUnavailable, Unavailable, UnavailableReason,
)

_API_BASE: Final = "https://ascend.wiki"
SEARCH_ENDPOINT: Final = Endpoint(f"{_API_BASE}/search")
_TIMEOUT: Final = Seconds(10.0)
_SEARCH_LIMIT: Final = ByteLimit(1_048_576)
_SKILL_LIMIT: Final = ByteLimit(262_144)
_BACKOFFS: Final = (Seconds(0.5), Seconds(1.0), Seconds(2.0))


class ProductionSleeper:
    def sleep(self, delay: Seconds) -> None:
        time.sleep(delay)


class AscendKgProvider:
    def __init__(
        self,
        transport: Transport,
        sleeper: Sleeper,
        api_key: str | None,
    ) -> None:
        self._transport = transport
        self._sleeper = sleeper
        normalized_key = api_key.strip() if api_key is not None else ""
        self._api_key = normalized_key or None
        self._api_key_is_header_safe = (
            not normalized_key
            or (normalized_key.isascii() and normalized_key.isprintable())
        )
        self._latest_response_token: ResponseToken | None = None

    @classmethod
    def from_environment(
        cls,
        transport: Transport,
        sleeper: Sleeper,
    ) -> AscendKgProvider:
        return cls(transport, sleeper, os.environ.get("ASCEND_KG_API_KEY"))

    def search(self, request: SearchRequest) -> SearchResult:
        match request.network_consent:
            case NetworkConsent.GRANTED:
                pass
            case NetworkConsent.NOT_REQUESTED | NetworkConsent.REFUSED:
                return InvalidResponse(InvalidResponseReason.CONSENT_REQUIRED)
            case unreachable:
                assert_never(unreachable)
        if self._api_key is None:
            return Unavailable(UnavailableReason.NO_API_KEY)
        if not self._api_key_is_header_safe:
            return Unavailable(UnavailableReason.CONFIGURATION)
        self._latest_response_token = None
        transport_result = self._send(
            TransportRequest(
                method=HttpMethod.POST,
                endpoint=SEARCH_ENDPOINT,
                headers=(
                    (HeaderName("X-API-Key"), HeaderValue(self._api_key)),
                    (HeaderName("Accept"), HeaderValue("application/json")),
                    (HeaderName("Content-Type"), HeaderValue("application/json")),
                ),
                body=request.prepared.body,
                timeout_seconds=_TIMEOUT,
                max_response_bytes=_SEARCH_LIMIT,
            ),
        )
        match transport_result:
            case TransportTimeout():
                return Unavailable(UnavailableReason.TIMEOUT)
            case TransportUnavailable():
                return Unavailable(UnavailableReason.SERVICE)
            case TransportResponse(status_code=200, body=body):
                result = parse_search_response(body)
                match result:
                    case Candidates(response_token=response_token):
                        self._latest_response_token = response_token
                    case NoMatch() | Ambiguous() | Unavailable() | InvalidResponse():
                        pass
                    case unreachable:
                        assert_never(unreachable)
                return result
            case TransportResponse(status_code=401 | 403):
                return Unavailable(UnavailableReason.CONFIGURATION)
            case TransportResponse(status_code=429):
                return Unavailable(UnavailableReason.RATE_LIMITED)
            case TransportResponse():
                return Unavailable(UnavailableReason.SERVICE)
            case unreachable:
                assert_never(unreachable)

    def load_skill(self, request: LoadRequest) -> RemoteLoadResult:
        match request.network_consent:
            case NetworkConsent.GRANTED:
                pass
            case NetworkConsent.NOT_REQUESTED | NetworkConsent.REFUSED:
                return RemoteLoadInvalid(InvalidResponseReason.CONSENT_REQUIRED)
            case unreachable:
                assert_never(unreachable)
        match request.activation_consent:
            case ActivationConsent.GRANTED:
                pass
            case ActivationConsent.NOT_REQUESTED | ActivationConsent.REFUSED:
                return RemoteLoadInvalid(InvalidResponseReason.CONSENT_REQUIRED)
            case unreachable:
                assert_never(unreachable)
        if self._api_key is None:
            return RemoteLoadUnavailable(UnavailableReason.NO_API_KEY)
        if not self._api_key_is_header_safe:
            return RemoteLoadUnavailable(UnavailableReason.CONFIGURATION)
        if request.response_token is not self._latest_response_token:
            return RemoteLoadInvalid(InvalidResponseReason.CANDIDATE_MEMBERSHIP)
        if request.candidate_id not in request.response_token.candidate_ids:
            return RemoteLoadInvalid(InvalidResponseReason.CANDIDATE_MEMBERSHIP)
        encoded_id = quote(str(request.candidate_id), safe="")
        transport_result = self._send(
            TransportRequest(
                method=HttpMethod.GET,
                endpoint=Endpoint(f"{_API_BASE}/skill/{encoded_id}"),
                headers=(
                    (HeaderName("X-API-Key"), HeaderValue(self._api_key)),
                    (HeaderName("Accept"), HeaderValue("text/markdown")),
                ),
                body=b"",
                timeout_seconds=_TIMEOUT,
                max_response_bytes=_SKILL_LIMIT,
            ),
        )
        match transport_result:
            case TransportTimeout():
                return RemoteLoadUnavailable(UnavailableReason.TIMEOUT)
            case TransportUnavailable():
                return RemoteLoadUnavailable(UnavailableReason.SERVICE)
            case TransportResponse(status_code=200, body=body):
                return parse_skill_response(request, body)
            case TransportResponse(status_code=401 | 403):
                return RemoteLoadUnavailable(UnavailableReason.CONFIGURATION)
            case TransportResponse(status_code=429):
                return RemoteLoadUnavailable(UnavailableReason.RATE_LIMITED)
            case TransportResponse():
                return RemoteLoadUnavailable(UnavailableReason.SERVICE)
            case unreachable:
                assert_never(unreachable)

    def _send(self, request: TransportRequest) -> TransportResult:
        for attempt in range(len(_BACKOFFS) + 1):
            result = self._transport.send(request)
            match result:
                case TransportResponse(status_code=429) if attempt < len(_BACKOFFS):
                    self._sleeper.sleep(_BACKOFFS[attempt])
                case TransportResponse() | TransportTimeout() | TransportUnavailable():
                    return result
                case unreachable:
                    assert_never(unreachable)
        raise AssertionError("retry loop must return")
