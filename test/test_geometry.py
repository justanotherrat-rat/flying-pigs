"""Pure-function checks: coordinate conversion, ship kinematics, Fresnel radius,
network metrics, and convoy generation. Expected values are derived by hand."""
import functools
import inspect

import numpy as np
import pytest

R = 6371000.0
M_PER_DEG_LAT = R * np.pi / 180.0          # 111194.93 m
KNOT = 0.514444


# ---------------------------------------------------------------- ENU <-> LLA
def test_lla_to_enu_is_zero_at_reference(lgb):
    e, n, u = lgb.lla_to_enu(1.4, 103.3, 0.0, 1.4, 103.3, 0.0)
    assert (e, n, u) == (0.0, 0.0, 0.0)


def test_one_degree_north_is_111_km(lgb):
    e, n, u = lgb.lla_to_enu(2.4, 103.3, 0.0, 1.4, 103.3, 0.0)
    assert e == pytest.approx(0.0, abs=1e-6)
    assert n == pytest.approx(M_PER_DEG_LAT, rel=1e-9)


def test_east_distance_scales_with_cos_latitude(lgb):
    e, _, _ = lgb.lla_to_enu(1.4, 104.3, 0.0, 1.4, 103.3, 0.0)
    assert e == pytest.approx(M_PER_DEG_LAT * np.cos(np.radians(1.4)), rel=1e-9)


def test_up_is_altitude_minus_reference(lgb):
    _, _, u = lgb.lla_to_enu(1.4, 103.3, 25.0, 1.4, 103.3, 5.0)
    assert u == pytest.approx(20.0)


def test_enu_lla_round_trip(lgb):
    rng = np.random.default_rng(0)
    for _ in range(50):
        e, n, u = rng.uniform(-60e3, 60e3, 2).tolist() + [rng.uniform(0, 6000)]
        lat, lon, alt = lgb.enu_to_lla(e, n, u, 1.4, 103.3, 0.0)
        e2, n2, u2 = lgb.lla_to_enu(lat, lon, alt, 1.4, 103.3, 0.0)
        assert (e2, n2, u2) == pytest.approx((e, n, u), abs=1e-6)


# ------------------------------------------------------------- ship movement
@functools.lru_cache(maxsize=1)
def _sea():
    from helpers import FakeDEM
    return FakeDEM(1.4, 103.3, half_deg=0.8)          # open sea, covers every move below


def _move(lgb, lat, lon, speed_kn, heading, minutes):
    """Call update_ship_positions_lla by keyword so the test survives the land-avoidance
    change (a dem_dataset parameter was inserted before time_step_minutes). If the
    function takes a DEM, it gets open sea, so these tests isolate pure kinematics."""
    lla = np.array([[lat, lon, 25.0]])
    params = inspect.signature(lgb.update_ship_positions_lla).parameters
    kw = {"time_step_minutes": minutes}
    dem_params = [n for n in params if "dem" in n.lower()]
    if dem_params:
        kw[dem_params[0]] = _sea()
    return lgb.update_ship_positions_lla(lla, np.array([speed_kn]), np.array([heading]), **kw)[0]


def test_ship_heading_north_moves_only_north(lgb):
    out = _move(lgb, 1.4, 103.3, 10, 0, 60)
    expected_dlat = 10 * KNOT * 3600 / M_PER_DEG_LAT        # 18.52 km
    assert out[0] - 1.4 == pytest.approx(expected_dlat, rel=1e-6)
    assert out[1] == pytest.approx(103.3, abs=1e-9)


def test_ship_heading_east_moves_only_east(lgb):
    out = _move(lgb, 1.4, 103.3, 10, 90, 60)
    expected_dlon = 10 * KNOT * 3600 / (M_PER_DEG_LAT * np.cos(np.radians(1.4)))
    assert out[1] - 103.3 == pytest.approx(expected_dlon, rel=1e-6)
    assert out[0] == pytest.approx(1.4, abs=1e-9)


@pytest.mark.parametrize("heading", [0, 45, 90, 135, 180, 270, 315])
def test_ship_one_hour_at_10kn_covers_18_52_km(lgb, heading):
    out = _move(lgb, 1.4, 103.3, 10, heading, 60)
    e, n, _ = lgb.lla_to_enu(out[0], out[1], 0.0, 1.4, 103.3, 0.0)
    assert np.hypot(e, n) == pytest.approx(10 * KNOT * 3600, rel=2e-3)


def test_ship_heading_is_compass_not_math_angle(lgb):
    se = _move(lgb, 1.4, 103.3, 10, 135, 30)    # compass 135 = south-east
    assert se[0] < 1.4 and se[1] > 103.3


def test_ship_altitude_and_zero_speed_are_preserved(lgb):
    out = _move(lgb, 1.4, 103.3, 0, 77, 60)
    assert tuple(out) == (1.4, 103.3, 25.0)


def test_ship_motion_is_additive_in_time(lgb):
    once = _move(lgb, 1.4, 103.3, 10, 60, 10)
    twice = _move(lgb, once[0], once[1], 10, 60, 10)
    both = _move(lgb, 1.4, 103.3, 10, 60, 20)
    assert twice[:2] == pytest.approx(both[:2], abs=1e-5)


def test_individual_movements_respect_spreads(lgb):
    np.random.seed(0)
    sp, hd = lgb.generate_individual_movements(speed=10, speed_spread=1, heading=135,
                                               heading_spread=30, n_ships=500)
    assert 9 <= sp.min() and sp.max() <= 11
    assert 105 <= hd.min() and hd.max() <= 165


# ------------------------------------------------------------ Fresnel radius
def test_fresnel_radius_midpoint_value(lgb):
    # r = sqrt(lambda * d1 * d2 / d); 10 km link at 1.5 GHz, midpoint
    lam = 299792458.0 / 1.5e9
    assert lgb.calculate_fresnel_radius(10000, 5000, 1.5) == pytest.approx(
        np.sqrt(lam * 5000 * 5000 / 10000), rel=1e-12)


def test_fresnel_radius_zero_at_endpoints(lgb):
    assert lgb.calculate_fresnel_radius(10000, 0, 1.5) == 0.0
    assert lgb.calculate_fresnel_radius(10000, 10000, 1.5) == 0.0


def test_fresnel_radius_symmetric_and_maximal_at_midpoint(lgb):
    r = [lgb.calculate_fresnel_radius(8000, x, 16.0) for x in np.linspace(0, 8000, 9)]
    assert r == pytest.approx(r[::-1])
    assert max(r) == r[4]


def test_fresnel_radius_scales_inverse_sqrt_frequency(lgb):
    ratio = (lgb.calculate_fresnel_radius(10000, 5000, 1.5)
             / lgb.calculate_fresnel_radius(10000, 5000, 16.0))
    assert ratio == pytest.approx(np.sqrt(16.0 / 1.5), rel=1e-12)


# ----------------------------------------------------------- network metrics
def _adj(n, edges):
    a = np.zeros((n, n))
    for i, j in edges:
        a[i, j] = a[j, i] = 1.0
    return a


def test_metrics_path_graph(lgb):
    m = lgb._network_metrics(_adj(3, [(0, 1), (1, 2)]))
    assert (m["components"], m["max_hops"], m["pairs_within_two_hops"],
            m["total_pairs"], m["direct_edges"]) == (1, 2, 3, 3, 2)


def test_metrics_two_components(lgb):
    m = lgb._network_metrics(_adj(4, [(0, 1), (2, 3)]))
    assert m["components"] == 2 and m["max_hops"] == 1
    assert m["pairs_within_two_hops"] == 2 and m["total_pairs"] == 6


def test_metrics_long_chain_exceeds_two_hops(lgb):
    m = lgb._network_metrics(_adj(5, [(0, 1), (1, 2), (2, 3), (3, 4)]))
    assert m["max_hops"] == 4
    assert m["pairs_within_two_hops"] == 4 + 3        # 4 pairs at 1 hop, 3 at 2 hops


def test_metrics_no_edges_is_infinite_hops(lgb):
    m = lgb._network_metrics(np.zeros((3, 3)))
    assert m["components"] == 3 and m["max_hops"] == np.inf and m["direct_edges"] == 0


def test_metrics_ignores_diagonal(lgb):
    a = _adj(2, [(0, 1)])
    np.fill_diagonal(a, 5.0)
    assert lgb._network_metrics(a)["direct_edges"] == 1


# ------------------------------------------------------------ convoy placement
def test_convoy_is_on_water_inside_dem_and_in_range(lgb, flat_dem):
    flat_dem.set_box_lonlat(103.0, 103.2, 1.0, 1.8, 40.0)      # land strip inside the area
    np.random.seed(3)
    ships = lgb.generate_convoy_lla(flat_dem, n_ships=8)
    assert ships.shape == (8, 3)
    for lat, lon, alt in ships:
        assert not lgb.is_on_land(lon, lat, flat_dem)
        assert 20.0 <= alt <= 30.0
        assert abs(lat - 1.4) <= 0.4 and abs(lon - 103.2) <= 0.4


def test_convoy_raises_when_no_water(lgb, flat_dem):
    flat_dem.data[:] = 10.0
    with pytest.raises(RuntimeError):
        lgb.generate_convoy_lla(flat_dem, n_ships=3, max_attempts_per_point=5)


def test_convoy_without_dem_falls_back_to_unrestricted(lgb):
    assert lgb.generate_convoy_lla(None, n_ships=4).shape == (4, 3)


def test_outside_dem_counts_as_land(lgb, flat_dem):
    assert lgb.is_on_land(10.0, 50.0, flat_dem) is True