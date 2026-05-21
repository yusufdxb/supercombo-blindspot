"""Gate-the-Step-4 sanity check: CARLA server is reachable at localhost:2000
and client + server versions match exactly. Run with:

    .venv/bin/python -m pytest tests/test_carla_connection.py -v

Requires CARLA server running:
    cd ~/Sim/CARLA_0.9.15 && ./CarlaUE4.sh -RenderOffScreen -quality-level=Epic -carla-rpc-port=2000
"""

from __future__ import annotations

import socket

import pytest

CARLA_HOST = "localhost"
CARLA_PORT = 2000
CLIENT_TIMEOUT_S = 10.0
EXPECTED_VERSION = "0.9.15"


def _carla_reachable(host: str = CARLA_HOST, port: int = CARLA_PORT) -> bool:
    """True if a CARLA RPC server is listening. Used to skip this integration
    suite cleanly when no simulator is running (fresh clone, CI)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


# Whole module is an integration suite: skip it unless a CARLA server answers,
# rather than hard-failing on every machine that isn't running the simulator.
pytestmark = pytest.mark.skipif(
    not _carla_reachable(),
    reason=f"no CARLA server on {CARLA_HOST}:{CARLA_PORT} — integration tests skipped",
)


@pytest.fixture(scope="module")
def carla_client():
    try:
        import carla
    except ImportError as e:
        pytest.skip(f"carla python package not installed: {e}")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CLIENT_TIMEOUT_S)
    return client


def test_port_open():
    """Confirm the CARLA RPC port accepts a TCP connection."""
    assert _carla_reachable(), f"nothing listening on {CARLA_HOST}:{CARLA_PORT}"


def test_versions_match(carla_client):
    server = carla_client.get_server_version()
    client = carla_client.get_client_version()
    assert server == client, f"version mismatch: server={server!r} client={client!r}"
    assert server == EXPECTED_VERSION, (
        f"unexpected CARLA version: got {server!r}, expected {EXPECTED_VERSION!r}"
    )


def test_world_reachable(carla_client):
    """Smoke-test a non-trivial RPC: load the world handle."""
    world = carla_client.get_world()
    assert world is not None
    # the map exists and has a sensible name
    map_name = world.get_map().name
    assert isinstance(map_name, str) and len(map_name) > 0


def test_available_maps(carla_client):
    """We need Town04 and Town06 specifically for the overpass survey."""
    maps = carla_client.get_available_maps()
    map_names = {m.rsplit("/", 1)[-1] for m in maps}
    assert "Town04" in map_names, f"Town04 missing from {sorted(map_names)}"
    assert "Town06" in map_names, f"Town06 missing from {sorted(map_names)}"
