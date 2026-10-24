"""Integration tests for map provider selection, OSM fallback, tile caching and zone safety.

What the code base actually has (and what is therefore tested here):

* the provider the UI reads is ``/api/v2/settings/maps`` (Google by default) -- the free-form
  ``ui.map_provider`` key of ``/api/v2/settings/system`` is stored but read by nothing;
* ``MapsService`` (tile cache, usage counters, per-tile Google -> OSM fallback, ``bypass_external``)
  exercised against a fake upstream so no request ever leaves the machine;
* zones (``/api/v2/map/zones``) and the stored map configuration are independent of the provider.

There is no code for adaptive throttling, cost estimation/thresholds, cache expiry or size bounds,
Google key permission/quota probing, or per-provider performance tracking, so nothing here
pretends to test them.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from backend.src.main import app
from backend.src.services import maps_service as maps_module
from backend.src.services.maps_service import MapsService

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
GOOGLE_TILE = PNG_MAGIC + b"google-satellite-tile"
OSM_TILE = PNG_MAGIC + b"osm-standard-tile"
GOOGLE_KEY = "AIza-integration-test-key"


# --------------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------------


class Upstream:
    """The network, as far as ``MapsService`` can see it: Google + OSM tile/geocode servers."""

    def __init__(self, google_failure: int | str | None = None, osm_failure: bool = False) -> None:
        self.google_failure = google_failure
        self.osm_failure = osm_failure
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        if host == "maps.googleapis.com":
            if self.google_failure == "connect_error":
                raise httpx.ConnectError("network unreachable", request=request)
            if self.google_failure is not None:
                return httpx.Response(int(self.google_failure), json={"status": "DENIED"})
            return httpx.Response(200, content=GOOGLE_TILE)
        if host == "tile.openstreetmap.org":
            if self.osm_failure:
                return httpx.Response(503)
            return httpx.Response(200, content=OSM_TILE)
        raise AssertionError(f"unexpected outbound request to {request.url}")


class _HttpxShim:
    """Stands in for the ``httpx`` name inside ``maps_service`` only (the test keeps real httpx)."""

    def __init__(self, upstream: Upstream) -> None:
        self._upstream = upstream

    def AsyncClient(self, *args, **kwargs):  # noqa: N802 - mirrors httpx.AsyncClient
        return httpx.AsyncClient(transport=httpx.MockTransport(self._upstream.handler))


@pytest.fixture
def upstream(monkeypatch) -> Upstream:
    fake = Upstream()
    monkeypatch.setattr(maps_module, "httpx", _HttpxShim(fake))
    return fake


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Private settings/map-config store; operator auth off like the rest of the suite."""
    monkeypatch.setenv("LAWNBERRY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "0")
    return tmp_path


@pytest_asyncio.fixture
async def client(data_dir):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest_asyncio.fixture
async def restore_zones(client):
    """``/api/v2/map/zones`` is a process-wide in-memory store: put back what the test found."""
    original = (await client.get("/api/v2/map/zones")).json()
    yield
    await client.post("/api/v2/map/zones", json=original)


# --------------------------------------------------------------------------------------------
# Defaults and provider switching (HTTP)
# --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_google_maps_default_provider(client):
    """A fresh install uses Google Maps, with no key, no bypass and no fallback in effect."""
    settings = await client.get("/api/v2/settings/maps")
    assert settings.status_code == 200
    assert settings.json() == {"provider": "google", "bypass_external": False, "api_key": None}

    # The stored map configuration agrees (it spells the provider "google-maps").
    config = (await client.get("/api/v2/map/configuration")).json()
    assert config["provider"] == "google-maps"
    assert config["fallback"] == {"active": False, "reason": None, "provider": "google-maps"}
    assert config["zones"] == []


@pytest.mark.asyncio
async def test_map_provider_switching_preserves_zones(client, data_dir, restore_zones):
    """Switching Google <-> OSM changes only the provider: zones and map config stay as saved."""
    zones = [
        {
            "id": "provider-test-zone",
            "name": "Provider Switch Test",
            "polygon": [
                {"latitude": 40.7128, "longitude": -74.0060},
                {"latitude": 40.7129, "longitude": -74.0060},
                {"latitude": 40.7129, "longitude": -74.0059},
                {"latitude": 40.7128, "longitude": -74.0059},
            ],
            "priority": 1,
            "exclusion_zone": False,
        },
        {
            "id": "provider-test-flowerbed",
            "name": "Flower bed",
            "polygon": [
                {"latitude": 40.71282, "longitude": -74.00598},
                {"latitude": 40.71284, "longitude": -74.00598},
                {"latitude": 40.71284, "longitude": -74.00596},
            ],
            "priority": 5,
            "exclusion_zone": True,
        },
    ]
    saved = await client.post("/api/v2/map/zones", json=zones)
    assert saved.status_code == 200 and saved.json() == zones

    map_config = {
        "zones": [
            {
                "zone_id": "yard",
                "zone_type": "boundary",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0.0003, 0], [0.0003, 0.0003], [0, 0.0003], [0, 0]]],
                },
            }
        ],
        "markers": {"home": {"lat": 40.7128, "lng": -74.006, "type": "home"}},
    }
    put = await client.put("/api/v2/map/configuration", json=map_config)
    assert put.status_code == 200
    config_file = data_dir / "map_configuration_v2.json"
    stored_config = config_file.read_bytes()

    for provider in ("osm", "google"):
        switch = await client.put("/api/v2/settings/maps", json={"provider": provider})
        assert switch.status_code == 200
        assert (await client.get("/api/v2/settings/maps")).json()["provider"] == provider

        assert (await client.get("/api/v2/map/zones")).json() == zones
        current = (await client.get("/api/v2/map/configuration")).json()
        assert current["zones"] == map_config["zones"]
        assert current["markers"] == map_config["markers"]
        assert config_file.read_bytes() == stored_config, "a provider switch must not rewrite it"


# --------------------------------------------------------------------------------------------
# MapsService: fallback, caching and bypass (fake upstream, no network)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("google_failure", [429, 403, "connect_error"])
@pytest.mark.asyncio
async def test_osm_fallback_when_google_unavailable(upstream, google_failure):
    """Quota exhausted (429), key rejected (403) or Google unreachable: the tile comes from OSM."""
    upstream.google_failure = google_failure
    service = MapsService(provider="google", api_key=GOOGLE_KEY)

    tile = await service.get_map_tile(18, 41000, 98000)

    assert tile == OSM_TILE, "map functionality must continue with the OSM tile"
    assert [r.url.host for r in upstream.requests] == [
        "maps.googleapis.com",
        "tile.openstreetmap.org",
    ]
    assert upstream.requests[0].url.params["key"] == GOOGLE_KEY
    assert upstream.requests[1].headers["user-agent"] == "LawnBerry-Pi/2.0", (
        "the OSM tile usage policy requires an identifying User-Agent"
    )
    stats = service.get_usage_stats()
    assert (stats["google_requests"], stats["osm_requests"], stats["failed_requests"]) == (0, 1, 1)


@pytest.mark.asyncio
async def test_osm_fallback_when_every_provider_fails(upstream):
    """With Google *and* OSM down the caller still gets decodable image bytes, not an exception."""
    upstream.google_failure = 429
    upstream.osm_failure = True
    service = MapsService(provider="google", api_key=GOOGLE_KEY)

    tile = await service.get_map_tile(18, 1, 2)

    assert tile.startswith(PNG_MAGIC) and tile not in (GOOGLE_TILE, OSM_TILE)
    assert [r.url.host for r in upstream.requests] == [
        "maps.googleapis.com",
        "tile.openstreetmap.org",
    ]
    assert service.get_usage_stats()["osm_requests"] == 0  # a failed fetch is not a served tile


@pytest.mark.asyncio
async def test_map_tile_caching(upstream):
    """A tile is fetched once per provider; hits are counted; clear_cache() forces a refetch."""
    service = MapsService(provider="google", api_key=GOOGLE_KEY)

    assert await service.get_map_tile(18, 10, 20) == GOOGLE_TILE
    assert await service.get_map_tile(18, 10, 20) == GOOGLE_TILE
    assert len(upstream.requests) == 1, "the second request must be served from the cache"
    stats = service.get_usage_stats()
    assert (stats["google_requests"], stats["cached_requests"], stats["cache_size"]) == (1, 1, 1)

    assert await service.get_map_tile(18, 11, 20) == GOOGLE_TILE  # another tile is a miss
    assert len(upstream.requests) == 2
    assert service.get_usage_stats()["cache_size"] == 2

    # The cache is per provider: the satellite tile must not be served for the OSM layer.
    service.configure("osm")
    assert await service.get_map_tile(18, 10, 20) == OSM_TILE
    assert upstream.requests[-1].url.host == "tile.openstreetmap.org"

    service.clear_cache()
    assert service.get_usage_stats()["cache_size"] == 0
    assert await service.get_map_tile(18, 10, 20) == OSM_TILE
    assert len(upstream.requests) == 4


@pytest.mark.asyncio
async def test_map_bypass_makes_no_external_requests(upstream):
    """bypass_external is the hard cost cap: offline tile, no geocoding, zero outbound calls."""
    service = MapsService(provider="google", api_key=GOOGLE_KEY, bypass_external=True)

    tile = await service.get_map_tile(18, 10, 20)

    assert tile.startswith(PNG_MAGIC) and tile not in (GOOGLE_TILE, OSM_TILE)
    assert await service.geocode("1 Main St") is None
    assert await service.reverse_geocode(40.7128, -74.006) is None
    assert upstream.requests == []
    stats = service.get_usage_stats()
    assert (stats["google_requests"], stats["osm_requests"], stats["cache_size"]) == (0, 0, 0)
