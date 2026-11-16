"""The ship<->ship graph cache in build_dual_band_graph / place_uavs must be a pure speed-up:
bit-identical graphs and identical placements, at a fraction of the cost."""
import time

import numpy as np
import pytest

LAT, LON = 1.4, 103.3


def _ships(rng, n, span=30000.0):
    return np.column_stack([rng.uniform(-span, span, n), rng.uniform(-span, span, n), rng.uniform(20, 30, n)])


def _terrain(dem):
    dem.set_box_enu(LAT, LON, -3000, 3000, -40000, 8000, 60.0)       # a ridge crossing many ship-ship links
    dem.set_box_enu(LAT, LON, 12000, 16000, -9000, 40000, 25.0)


@pytest.mark.parametrize("seed", range(4))
def test_cached_graphs_are_identical_to_full_graphs(lgb, flat_dem, seed):
    _terrain(flat_dem)
    rng = np.random.default_rng(seed)
    ships = _ships(rng, 8)
    ship_l = lgb.build_dual_band_graph(ships, flat_dem, LAT, LON, n_ships=len(ships))[0]
    for _ in range(5):
        uav = np.array([[*rng.uniform(-30000, 30000, 2), rng.uniform(1500, 6100)]])
        coords = np.vstack((ships, uav))
        full_l, full_ku = lgb.build_dual_band_graph(coords, flat_dem, LAT, LON, n_ships=len(ships))
        fast_l, fast_ku = lgb.build_dual_band_graph(coords, flat_dem, LAT, LON, n_ships=len(ships),
                                                    ship_graph_l=ship_l)
        assert np.array_equal(full_l.toarray(), fast_l.toarray())
        assert np.array_equal(full_ku.toarray(), fast_ku.toarray())


def test_cache_accepts_dense_array_and_rejects_wrong_shape(lgb, flat_dem):
    ships = _ships(np.random.default_rng(0), 4)
    coords = np.vstack((ships, [[0, 0, 3000.0]]))
    ship_l = lgb.build_dual_band_graph(ships, flat_dem, LAT, LON, n_ships=4)[0]
    a = lgb.build_dual_band_graph(coords, flat_dem, LAT, LON, n_ships=4, ship_graph_l=ship_l)[0]
    b = lgb.build_dual_band_graph(coords, flat_dem, LAT, LON, n_ships=4, ship_graph_l=ship_l.toarray())[0]
    assert np.array_equal(a.toarray(), b.toarray())
    with pytest.raises(ValueError):
        lgb.build_dual_band_graph(coords, flat_dem, LAT, LON, n_ships=4, ship_graph_l=np.zeros((3, 3)))


def test_place_uavs_gives_the_same_placement_with_and_without_the_cache(lgb, flat_dem, monkeypatch):
    _terrain(flat_dem)
    ships = _ships(np.random.default_rng(3), 6, span=25000.0)
    cached = lgb.place_uavs(ships, flat_dem, LAT, LON, time_budget_s=120.0, random_state=7)
    orig = lgb.build_dual_band_graph
    monkeypatch.setattr(lgb, "build_dual_band_graph",
                        lambda c, d, la, lo, n_ships, ship_graph_l=None: orig(c, d, la, lo, n_ships))
    plain = lgb.place_uavs(ships, flat_dem, LAT, LON, time_budget_s=120.0, random_state=7)
    assert not cached["timed_out"] and not plain["timed_out"]
    assert cached["uavs"][0]["uav_enu"] == pytest.approx(plain["uavs"][0]["uav_enu"], abs=1e-9)
    assert np.array_equal(cached["augmented_graph"].toarray(), plain["augmented_graph"].toarray())
    assert cached["network_metrics"] == plain["network_metrics"]


@pytest.mark.slow
def test_thirty_ship_candidate_evaluation_is_much_cheaper(lgb, real_dem):
    """On the real DEM at 30 ships, evaluating a candidate with the cache must be >= 5x faster
    (435 of 465 pairs are skipped)."""
    rng = np.random.default_rng(0)
    ships = []
    while len(ships) < 30:
        e, n = rng.uniform(-30000, 30000, 2)
        lat, lon, _ = lgb.enu_to_lla(e, n, 0.0, LAT, LON, 0.0)
        if not lgb.is_on_land(lon, lat, real_dem):
            ships.append([e, n, rng.uniform(20, 30)])
    ships = np.array(ships)
    coords = np.vstack((ships, [[0.0, 0.0, 3000.0]]))
    ship_l = lgb.build_dual_band_graph(ships, real_dem, LAT, LON, n_ships=30)[0]

    def timed(**kw):
        t = time.perf_counter()
        for _ in range(3):
            lgb.build_dual_band_graph(coords, real_dem, LAT, LON, n_ships=30, **kw)
        return (time.perf_counter() - t) / 3

    slow, fast = timed(), timed(ship_graph_l=ship_l)
    print(f"\n30 ships: full {slow * 1000:.0f} ms vs cached {fast * 1000:.0f} ms per candidate = {slow / fast:.1f}x")
    assert slow / fast >= 5.0, f"full {slow * 1000:.0f} ms vs cached {fast * 1000:.0f} ms"


# ------------------------------------------------------------------ in-memory DEM cache
@pytest.mark.slow
def test_dem_cache_returns_exactly_what_per_pixel_reads_return(lgb, real_dem):
    """Real DEM, 3000 random points incl. outside coverage: cached lookup == a direct 1x1 read."""
    import rasterio
    rng = np.random.default_rng(1)
    b = real_dem.bounds
    for lon, lat in zip(rng.uniform(b.left - 0.1, b.right + 0.1, 3000), rng.uniform(b.bottom - 0.1, b.top + 0.1, 3000)):
        got = lgb.get_elevation_at_lonlat(lon, lat, real_dem)
        row, col = real_dem.index(lon, lat)
        if row < 0 or col < 0 or row >= real_dem.height or col >= real_dem.width:
            assert got is None
        else:
            direct = float(real_dem.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])
            assert got == direct


def test_dem_cache_sees_edits_made_to_an_in_memory_dem(lgb, flat_dem):
    lon, lat = 103.3, 1.4
    assert lgb.get_elevation_at_lonlat(lon, lat, flat_dem) == 0.0        # first call fills the cache
    flat_dem.set_box_lonlat(103.29, 103.31, 1.39, 1.41, 77.0)
    assert lgb.get_elevation_at_lonlat(lon, lat, flat_dem) == 77.0


def test_dem_cache_falls_back_when_read_cannot_return_the_whole_band(lgb):
    class WindowOnly:                       # a dataset whose read() demands a window
        height = width = 10
        nodata = None
        def index(self, lon, lat): return (int(lat), int(lon))
        def read(self, band, window):
            return np.full((1, 1), 5.0 + window.col_off)
    assert lgb.get_elevation_at_lonlat(3.2, 4.7, WindowOnly()) == 8.0
