"""End-to-end invariants of main(). These guard bugs that were found and fixed:

  * ships must never sail onto land (original code had no land check:
    ~68% of random 60-minute runs put a ship on land, median at minute 28)
  * the Fresnel clearance test must include Earth curvature (original code
    accepted 25 m-antenna ship-ship links to ~34 km over open sea; with a
    4/3-Earth bulge the 60% Fresnel zone is only clear to ~20 km)

They are EXPECTED TO FAIL on code that lacks those fixes - run them against
your fixed copy with:   pytest --lgb-module=path/to/losGenerationBands   (or LGB_MODULE=...)
"""
import csv

import numpy as np
import pytest

from helpers import write_geotiff

pytestmark = pytest.mark.scenario


def ships_by_step(lgb, nodes_csv, ref_lat, ref_lon, n_ships):
    with open(nodes_csv) as f:
        rows = [r for r in csv.DictReader(f) if int(r["node"]) < n_ships]
    return [(int(r["step"]), int(r["node"]),
             *lgb.enu_to_lla(float(r["x"]), float(r["y"]), float(r["z"]), ref_lat, ref_lon, 0.0)[:2])
            for r in rows]                                   # (step, ship, lat, lon)


def run_and_check_water(lgb, monkeypatch, tmp_path, dem_path, seed, fixed_ships=None, n_ships=4):
    import rasterio
    captured = {}
    orig = lgb.generate_convoy_lla

    def wrapper(dem, n_ships=n_ships, **kw):
        out = fixed_ships.copy() if fixed_ships is not None else orig(dem, n_ships=n_ships, **kw)
        captured["ref"] = (out[0, 0], out[0, 1])             # ENU reference = ship 0 start
        return out

    monkeypatch.setattr(lgb, "generate_convoy_lla", wrapper)
    np.random.seed(seed)
    lgb.main(sim_time=60, calc_interval=2, n_ships=n_ships, placement_mode="midpoint",
             make_plots=False, dem_file_path=str(dem_path), export_dir=str(tmp_path / "exp"),
             plot_dir=str(tmp_path / "plots"))
    ref_lat, ref_lon = captured["ref"]
    track = ships_by_step(lgb, tmp_path / "exp" / "nodes.csv", ref_lat, ref_lon, n_ships)
    assert track, "no ship rows exported"
    with rasterio.open(dem_path) as dem:
        return [(s, n, lat, lon) for s, n, lat, lon in track if lgb.is_on_land(lon, lat, dem)]


def test_ships_never_enter_a_landmass_in_their_path(lgb, monkeypatch, tmp_path):
    """Synthetic world: open sea west of lon 103.33, land everywhere east of it.
    Ships start in the west and head south-east, straight at the coast."""
    tif = write_geotiff(tmp_path / "coast.tif", 1.4, 103.3, half_deg=0.5, land_east_of_lon=103.33)
    start = np.array([[1.400, 103.300, 25.], [1.410, 103.302, 25.],
                      [1.395, 103.298, 25.], [1.405, 103.305, 25.]])
    bad = run_and_check_water(lgb, monkeypatch, tmp_path, tif, seed=0, fixed_ships=start)
    assert not bad, f"{len(bad)} ship positions on land, first: {bad[:3]}"


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(6))
def test_ships_stay_on_water_in_real_dem(lgb, monkeypatch, tmp_path, real_dem_path, seed):
    bad = run_and_check_water(lgb, monkeypatch, tmp_path, real_dem_path, seed)
    assert not bad, f"seed {seed}: {len(bad)} ship positions on land, first: {bad[:3]}"


# ------------------------------------------------------------ Earth curvature
def _clear(lgb, dem, ref, d_m, h, f=1.5):
    return lgb.check_fresnel_obstruction(np.array([0, 0, h], float), np.array([d_m, 0, h], float),
                                         ref[0], ref[1], dem, f)


def test_short_open_sea_link_is_clear(lgb, flat_dem, ref):
    assert _clear(lgb, flat_dem, ref, 10_000, 25)


def test_curvature_rejects_ship_ship_links_beyond_60pct_fresnel_range(lgb, flat_dem, ref):
    """25 m antennas, 1.5 GHz, open sea. Hand calculation of 0.6*F1 + bulge <= 25 m:
        k=1   -> ~18.6 km,   k=4/3 -> ~20.1 km.   Any sane treatment rejects 27 km.
    (Without curvature the code accepts up to ~34.5 km.)"""
    assert not _clear(lgb, flat_dem, ref, 27_000, 25)


def test_clearance_is_monotonic_in_distance_over_open_sea(lgb, flat_dem, ref):
    verdicts = [_clear(lgb, flat_dem, ref, d, 25) for d in range(2_000, 40_001, 2_000)]
    first_block = verdicts.index(False)
    assert not any(verdicts[first_block:]), f"clear again after blocked: {verdicts}"


def test_taller_antennas_reach_further(lgb, flat_dem, ref):
    def reach(h):
        return max((d for d in range(2_000, 45_001, 500) if _clear(lgb, flat_dem, ref, d, h)),
                   default=0)
    assert reach(20) < reach(25) < reach(30)


def test_los_graph_drops_30km_ship_pair_over_open_sea(lgb, flat_dem, ref):
    c = np.array([[0, 0, 25.], [30_000, 0, 25.]])
    assert lgb.build_los_graph(c, flat_dem, ref[0], ref[1])[0, 1] == 0
