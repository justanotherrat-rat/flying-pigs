"""UAV placement: objective ordering, constraints, determinism, and agreement
with an exhaustive grid search (on flat sea so evaluations are fast)."""
import itertools

import numpy as np
import pytest

from helpers import placement_score

LAT, LON = 1.4, 103.3
# Three ships in a row, plus one 60 km north - beyond ship-to-ship radio horizon,
# so ships alone are disconnected and only a UAV can join the network.
SHIPS = np.array([[0, 0, 25.], [15_000, 0, 25.], [-15_000, 0, 25.], [0, 60_000, 25.]])


def place(lgb, dem, **kw):
    kw.setdefault("time_budget_s", 60.0)
    kw.setdefault("random_state", 7)
    return lgb.place_uavs(SHIPS, dem, LAT, LON, **kw)


def uav_link_km(res):
    return float(np.sum(res["augmented_graph"].toarray()[-1, :len(SHIPS)])) / 1000.0


def test_ships_alone_are_disconnected(lgb, flat_dem):
    adj = lgb.build_los_graph(SHIPS, flat_dem, LAT, LON).toarray()
    assert lgb._network_metrics(adj)["components"] > 1


def test_uav_connects_the_whole_network(lgb, flat_dem):
    res = place(lgb, flat_dem)
    assert res["network_metrics"]["components"] == 1
    assert res["feasible_with_max_uavs"] is True
    assert sorted(res["visible_ships"]) == [0, 1, 2, 3]
    assert not res["timed_out"]


def test_uav_stays_inside_search_bounds(lgb, flat_dem):
    res = place(lgb, flat_dem, min_alt_m=1500, max_alt_m=6100)
    e, n, u = res["uavs"][0]["uav_enu"]
    span = np.maximum(SHIPS[:, :2].max(0) - SHIPS[:, :2].min(0), 1000.0)
    lo, hi = SHIPS[:, :2].min(0) - 0.15 * span, SHIPS[:, :2].max(0) + 0.15 * span
    assert lo[0] <= e <= hi[0] and lo[1] <= n <= hi[1]
    assert 1500 <= u <= 6100


def test_enu_and_lla_outputs_agree(lgb, flat_dem):
    res = place(lgb, flat_dem)
    uav = res["uavs"][0]
    lat, lon, alt = lgb.enu_to_lla(*uav["uav_enu"], LAT, LON, 0.0)
    assert uav["uav_lla"] == pytest.approx([lat, lon, alt])


def test_midpoint_mode_returns_centroid_at_mid_altitude(lgb, flat_dem):
    """Midpoint = ship centroid in x/y and the middle of the altitude range in z.
    (Earlier versions seeded at min_alt_m; the current code uses (min+max)/2.)"""
    res = place(lgb, flat_dem, optimize=False, min_alt_m=1500, max_alt_m=6100)
    e, n, u = res["uavs"][0]["uav_enu"]
    assert (e, n) == pytest.approx((SHIPS[:, 0].mean(), SHIPS[:, 1].mean()))
    assert u == pytest.approx((1500 + 6100) / 2)


def test_optimizer_is_never_worse_than_midpoint(lgb, flat_dem):
    mid = place(lgb, flat_dem, optimize=False)
    opt = place(lgb, flat_dem)
    s_mid = placement_score(lgb, mid["augmented_graph"].toarray())
    s_opt = placement_score(lgb, opt["augmented_graph"].toarray())
    assert s_opt <= s_mid
    if s_opt == s_mid:                       # tie on connectivity -> shorter links win
        assert uav_link_km(opt) <= uav_link_km(mid) + 1e-9


def test_same_seed_is_deterministic(lgb, flat_dem):
    a, b = place(lgb, flat_dem), place(lgb, flat_dem)
    assert a["uavs"][0]["uav_enu"] == pytest.approx(b["uavs"][0]["uav_enu"], abs=1e-9)


def test_optimizer_matches_or_beats_coarse_grid(lgb, flat_dem):
    """Independent oracle (same idea as testing/directPlacementTest.py)."""
    res = place(lgb, flat_dem, min_alt_m=1500, max_alt_m=6100)
    lo = SHIPS[:, :2].min(0) - 0.15 * (SHIPS[:, :2].max(0) - SHIPS[:, :2].min(0))
    hi = SHIPS[:, :2].max(0) + 0.15 * (SHIPS[:, :2].max(0) - SHIPS[:, :2].min(0))
    best = np.inf
    for e, n, u in itertools.product(np.linspace(lo[0], hi[0], 5), np.linspace(lo[1], hi[1], 5),
                                     np.linspace(1500, 6100, 3)):
        g_l, g_ku = lgb.build_dual_band_graph(np.vstack((SHIPS, [e, n, u])), flat_dem, LAT, LON,
                                              n_ships=len(SHIPS))
        best = min(best, placement_score(lgb, g_l.maximum(g_ku).toarray()))
    assert placement_score(lgb, res["augmented_graph"].toarray()) <= best


# -------------------------------------------------------------- travel limit
def test_first_placement_has_no_travel_constraint(lgb, flat_dem):
    res = place(lgb, flat_dem)
    assert res["travel_distance_m"] is None and res["travel_feasible"] is True


def test_travel_limit_is_respected_when_binding(lgb, flat_dem):
    prev = np.array([19_000.0, 0.0, 1500.0])             # far from the ideal spot
    res = place(lgb, flat_dem, previous_uav_enu=prev, travel_time_minutes=5, uav_speed_kmh=130)
    limit = 130 * 1000 * 5 / 60                           # 10 833 m
    assert res["max_travel_distance_m"] == pytest.approx(limit)
    assert res["travel_feasible"] is True
    assert res["travel_distance_m"] <= limit + 1e-6
    assert np.linalg.norm(res["uavs"][0]["uav_enu"] - prev) == pytest.approx(
        res["travel_distance_m"])


def test_longer_interval_allows_longer_travel(lgb):
    # pure arithmetic of the limit, no optimisation needed
    from helpers import FakeDEM
    dem = FakeDEM(LAT, LON, half_deg=0.8)
    prev = np.array([0.0, 0.0, 1500.0])
    short = place(lgb, dem, optimize=False, previous_uav_enu=prev, travel_time_minutes=1,
                  uav_speed_kmh=120)
    long_ = place(lgb, dem, optimize=False, previous_uav_enu=prev, travel_time_minutes=10,
                  uav_speed_kmh=120)
    assert short["max_travel_distance_m"] == pytest.approx(2000.0)
    assert long_["max_travel_distance_m"] == pytest.approx(20000.0)


# ---------------------------------------------------------------- guard rails
@pytest.mark.parametrize("kw", [
    {"uav_speed_kmh": 0}, {"uav_speed_kmh": -5}, {"uav_speed_kmh": float("nan")},
    {"min_alt_m": 7000, "max_alt_m": 6100}, {"min_alt_m": -1}, {"max_alt_m": float("inf")},
])
def test_invalid_numeric_arguments_raise(lgb, flat_dem, kw):
    with pytest.raises(ValueError):
        place(lgb, flat_dem, optimize=False, **kw)


def test_bad_previous_position_shape_raises(lgb, flat_dem):
    with pytest.raises(ValueError):
        place(lgb, flat_dem, previous_uav_enu=[0, 0], travel_time_minutes=5)


def test_previous_position_requires_travel_time(lgb, flat_dem):
    with pytest.raises(ValueError):
        place(lgb, flat_dem, previous_uav_enu=[0, 0, 1500])


def test_bad_ship_array_shapes_raise(lgb, flat_dem):
    with pytest.raises(ValueError):
        lgb.place_uavs(np.zeros((3, 2)), flat_dem, LAT, LON)
    with pytest.raises(ValueError):
        lgb.place_uavs(np.zeros((0, 3)), flat_dem, LAT, LON)


def test_zero_budget_raises_timeout_instead_of_returning_garbage(lgb, flat_dem):
    with pytest.raises(TimeoutError):
        place(lgb, flat_dem, time_budget_s=0.0)


def test_tiny_budget_still_returns_a_valid_flagged_result(lgb, flat_dem):
    res = place(lgb, flat_dem, time_budget_s=0.05)
    assert res["timed_out"] is True
    assert res["uavs"][0]["uav_enu"].shape == (3,)


@pytest.mark.slow
def test_default_budget_does_not_truncate_real_terrain(lgb, real_dem):
    """Guards finding: default 10 s budget must not silently cut the optimiser short
    (results would then depend on CPU speed)."""
    ships = np.array([[0, 0, 25.], [9000, 4000, 25.], [-8000, 7000, 25.], [3000, -9000, 25.]])
    res = lgb.place_uavs(ships, real_dem, 0.6, 103.45, time_budget_s=10.0, random_state=7)
    assert res["timed_out"] is False