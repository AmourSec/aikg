from __future__ import annotations

from collections import deque

from runtime.contracts import (
    LoadRequest,
    RemoteLoadResult,
    SearchRequest,
    SearchResult,
    Seconds,
    TransportRequest,
    TransportResult,
)


class FakeProvider:
    def __init__(
        self,
        search_results: tuple[SearchResult, ...],
        load_results: tuple[RemoteLoadResult, ...],
    ) -> None:
        self._search_results = deque(search_results)
        self._load_results = deque(load_results)
        self._search_requests: list[SearchRequest] = []
        self._load_requests: list[LoadRequest] = []

    @property
    def search_requests(self) -> tuple[SearchRequest, ...]:
        return tuple(self._search_requests)

    @property
    def load_requests(self) -> tuple[LoadRequest, ...]:
        return tuple(self._load_requests)

    def search(self, request: SearchRequest) -> SearchResult:
        self._search_requests.append(request)
        return self._search_results.popleft()

    def load_skill(self, request: LoadRequest) -> RemoteLoadResult:
        self._load_requests.append(request)
        return self._load_results.popleft()


class FakeTransport:
    def __init__(self, responses: tuple[TransportResult, ...]) -> None:
        self._responses = deque(responses)
        self._requests: list[TransportRequest] = []

    @property
    def requests(self) -> tuple[TransportRequest, ...]:
        return tuple(self._requests)

    def send(self, request: TransportRequest) -> TransportResult:
        self._requests.append(request)
        return self._responses.popleft()


class FakeSleeper:
    def __init__(self) -> None:
        self._delays: list[Seconds] = []

    @property
    def delays(self) -> tuple[Seconds, ...]:
        return tuple(self._delays)

    def sleep(self, delay: Seconds) -> None:
        self._delays.append(delay)
