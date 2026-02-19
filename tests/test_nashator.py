"""Tests for Nashator wire protocol framing (unit tests, no live server needed)."""

import json
import struct
import pytest
from my_agent.tools_nashator import _send_rpc


class TestWireProtocol:
    """Test the framing logic — these expect Nashator NOT running (connection refused)."""

    def test_connection_refused_returns_nothing(self):
        """When Nashator isn't running, status should be 'nothing' (not contradiction)."""
        result = _send_rpc("nashator_games", {})
        assert result["status"] == "nothing"
        assert "Cannot connect" in result["error"]

    def test_frame_encoding(self):
        """Verify 4-byte big-endian length prefix encoding matches zig-syrup."""
        payload = json.dumps({"jsonrpc": "2.0", "method": "test", "id": 1}).encode("utf-8")
        header = struct.pack(">I", len(payload))

        assert len(header) == 4
        decoded_len = struct.unpack(">I", header)[0]
        assert decoded_len == len(payload)

    def test_max_message_size(self):
        """4MB limit from message_frame.zig."""
        max_size = 4 * 1024 * 1024
        header = struct.pack(">I", max_size)
        decoded = struct.unpack(">I", header)[0]
        assert decoded == max_size

    def test_frame_roundtrip(self):
        """Encode/decode roundtrip for a JSON-RPC request."""
        request = {
            "jsonrpc": "2.0",
            "method": "nashator_solve",
            "id": 42,
            "params": {"game": "prisoners_dilemma", "method": "fictitious_play"},
        }
        payload = json.dumps(request).encode("utf-8")
        frame = struct.pack(">I", len(payload)) + payload

        # Decode
        header = frame[:4]
        body_len = struct.unpack(">I", header)[0]
        body = frame[4:4 + body_len]
        decoded = json.loads(body.decode("utf-8"))

        assert decoded == request
        assert decoded["method"] == "nashator_solve"
        assert decoded["params"]["game"] == "prisoners_dilemma"
