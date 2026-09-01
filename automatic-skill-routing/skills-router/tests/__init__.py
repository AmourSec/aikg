"""Suite-wide initialization for the skills-router unittest suite.

Importing this package (done by unittest discovery before any test module)
installs a guard that replaces ``socket.socket.connect`` with a stub that
raises immediately. Every test in the discovered suite therefore runs
offline: any code path that would open a real connection fails loudly at
the connect call instead of reaching the network.
"""

from __future__ import annotations

import socket
from typing import TypeAlias

__all__ = ["SocketConnectBlocked", "blocked_connect"]

SocketAddress: TypeAlias = (
    str
    | bytes
    | tuple[str, int]
    | tuple[str, int, int]
    | tuple[str, int, int, int]
)


class SocketConnectBlocked(RuntimeError):
    """Raised when a test tries to open a real socket connection."""


def blocked_connect(sock: socket.socket, address: SocketAddress) -> None:
    raise SocketConnectBlocked(
        f"test suite is offline: socket connect to {address!r} is blocked"
    )


socket.socket.connect = blocked_connect
