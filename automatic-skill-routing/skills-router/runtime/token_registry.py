from __future__ import annotations

import secrets
from typing import Protocol

from .contracts import ExternalResponseToken, ResponseToken


class TokenIssuer(Protocol):
    def __call__(self) -> str: ...


def _issue_external_response_token() -> str:
    return secrets.token_urlsafe(24)


class RegistryClosedError(RuntimeError):
    pass


class ResponseTokenRegistry:
    """Keep task handles mutable until the documented `_closed` terminal state."""

    __slots__ = ("_closed", "_issuer", "_tokens")

    def __init__(self, issuer: TokenIssuer = _issue_external_response_token) -> None:
        self._closed = False
        self._issuer = issuer
        self._tokens: dict[ExternalResponseToken, ResponseToken] = {}

    def issue(self, token: ResponseToken) -> ExternalResponseToken:
        if self._closed:
            raise RegistryClosedError
        handle = ExternalResponseToken(self._issuer())
        while handle in self._tokens:
            handle = ExternalResponseToken(self._issuer())
        self._tokens[handle] = token
        return handle

    def resolve(self, handle: ExternalResponseToken) -> ResponseToken | None:
        if self._closed:
            return None
        return self._tokens.get(handle)

    def close(self) -> None:
        self._tokens.clear()
        self._closed = True
