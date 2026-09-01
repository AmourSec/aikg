from __future__ import annotations

import socket
import unittest

from tests import SocketConnectBlocked, blocked_connect


class SocketGuardTests(unittest.TestCase):
    def test_discovery_installs_the_guard_suite_wide(self) -> None:
        # Given
        # unittest discovery imports the tests package before any test
        # module, so the guard is already live when this test runs.

        # When
        installed_connect = socket.socket.connect

        # Then
        self.assertIs(installed_connect, blocked_connect)

    def test_connect_raises_before_any_real_connection(self) -> None:
        # Given
        blocked = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(blocked.close)
        # Loopback port 1 is closed: without the guard this call raises
        # ConnectionRefusedError, so the guard error proves the raise
        # happens before any connection is attempted.

        # When
        # Then
        with self.assertRaises(SocketConnectBlocked):
            blocked.connect(("127.0.0.1", 1))


if __name__ == "__main__":
    unittest.main()
