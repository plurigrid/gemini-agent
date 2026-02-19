"""Nashator JSON-RPC bridge — connects ADK agents to the game theory solver.

Wire protocol: 4-byte big-endian length prefix + JSON-RPC 2.0 over TCP.
Default endpoint: 127.0.0.1:9999

Methods: nashator_solve, nashator_compose, nashator_gf3_check, nashator_games
"""

import json
import os
import socket
import struct


NASHATOR_HOST = os.environ.get("NASHATOR_HOST", "127.0.0.1")
NASHATOR_PORT = int(os.environ.get("NASHATOR_PORT", "9999"))
TIMEOUT = 10.0


def _send_rpc(method: str, params: dict, req_id: int = 1) -> dict:
    """Send a length-prefixed JSON-RPC 2.0 request to Nashator."""
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "id": req_id,
        "params": params,
    }
    payload = json.dumps(request).encode("utf-8")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        sock.connect((NASHATOR_HOST, NASHATOR_PORT))

        # 4-byte big-endian length prefix
        header = struct.pack(">I", len(payload))
        sock.sendall(header + payload)

        # Read response header
        resp_header = b""
        while len(resp_header) < 4:
            chunk = sock.recv(4 - len(resp_header))
            if not chunk:
                return {"error": "Connection closed reading header", "status": "contradiction"}
            resp_header += chunk

        resp_len = struct.unpack(">I", resp_header)[0]
        if resp_len > 4 * 1024 * 1024:
            return {"error": f"Response too large: {resp_len}", "status": "contradiction"}

        # Read response payload
        resp_payload = b""
        while len(resp_payload) < resp_len:
            chunk = sock.recv(resp_len - len(resp_payload))
            if not chunk:
                break
            resp_payload += chunk

        sock.close()

        response = json.loads(resp_payload.decode("utf-8"))
        if "error" in response:
            return {
                "error": response["error"],
                "status": "contradiction",
            }
        result = response.get("result", {})
        result["status"] = "value"
        return result

    except ConnectionRefusedError:
        return {
            "error": f"Cannot connect to Nashator at {NASHATOR_HOST}:{NASHATOR_PORT}. Is it running?",
            "status": "nothing",
        }
    except socket.timeout:
        return {"error": "Nashator request timed out", "status": "contradiction"}
    except Exception as e:
        return {"error": str(e), "status": "contradiction"}


# -- ADK tool functions -------------------------------------------------------

def nashator_solve(
    game: str,
    method: str = "fictitious_play",
    max_iterations: int = 2000,
    epsilon: float = 0.02,
) -> dict:
    """Solve a game for Nash equilibrium via Nashator.

    Args:
        game: Game name (e.g. 'prisoners_dilemma', 'eip1559', 'gpu_routing')
              or 'custom' to provide payoffs directly.
        method: Solver method — 'gradient', 'fictitious_play', 'replicator', or 'propagator'.
        max_iterations: Maximum solver iterations.
        epsilon: Convergence threshold.
    """
    return _send_rpc("nashator_solve", {
        "game": game,
        "method": method,
        "maxIterations": max_iterations,
        "epsilon": epsilon,
    })


def nashator_solve_custom(
    payoffs: str,
    player_names: str,
    player_trits: str,
    method: str = "fictitious_play",
) -> dict:
    """Solve a custom game with explicit payoff tensor.

    Args:
        payoffs: JSON 3D array of payoffs, e.g. '[[[3,3],[0,5]],[[5,0],[1,1]]]'.
        player_names: JSON array of player name strings.
        player_trits: JSON array of GF(3) trit values (-1, 0, or 1).
        method: Solver method.
    """
    try:
        p = json.loads(payoffs)
        names = json.loads(player_names)
        trits = json.loads(player_trits)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}", "status": "contradiction"}

    return _send_rpc("nashator_solve", {
        "payoffs": p,
        "player_names": names,
        "player_trits": trits,
        "method": method,
    })


def nashator_compose(left: str, right: str, op: str = "seq") -> dict:
    """Compose two games with sequential or parallel operators.

    Args:
        left: Left game name.
        right: Right game name.
        op: Composition operator — 'seq' or 'par'.
    """
    return _send_rpc("nashator_compose", {
        "left": left,
        "right": right,
        "op": op,
    })


def nashator_gf3_check(trits: str) -> dict:
    """Validate GF(3) conservation over a trit sequence.

    Args:
        trits: JSON array of trit values (-1, 0, or 1), e.g. '[1, -1, 0]'.
    """
    try:
        t = json.loads(trits)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}", "status": "contradiction"}

    return _send_rpc("nashator_gf3_check", {"trits": t})


def nashator_games() -> dict:
    """List all available games in Nashator.

    Returns game names, player counts, trits, and types.
    """
    return _send_rpc("nashator_games", {})
