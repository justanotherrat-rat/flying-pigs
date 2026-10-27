"""Line-of-sight / Fresnel / dual-band graph behaviour on synthetic terrain
where the right answer is known by construction."""
import numpy as np
import pytest

REF_LAT, REF_LON = 1.4, 103.3


def fresnel_clear(lgb, dem, p1, p2, f):
    return lgb.check_fresnel_obstruction(np.array(p1, float), np.array(p2, float),
                                         REF_LAT, REF_LON, dem, f)


# ---------------------------------------------------------- Fresnel obstruction
def test_open_sea_short_link_is_clear_both_bands(lgb, flat_dem):
    for f in (lgb.FREQ_L_GHZ, lgb.FREQ_KU_GHZ):
        assert fresnel_clear(lgb, flat_dem, (0, 0, 25), (10_000, 0, 25), f)


def test_missing_dem_means_not_clear(lgb):
    assert not lgb.check_fresnel_obstruction(np.array([0, 0, 25.]), np.array([5000, 0, 25.]),
                                             REF_LAT, REF_LON, None, 1.5)


def test_link_leaving_dem_coverage_is_treated_as_blocked(lgb):
    from helpers import FakeDEM
    small = FakeDEM(REF_LAT, REF_LON, half_deg=0.05)             # ~5.5 km half-width
    assert not fresnel_clear(lgb, small, (0, 0, 25), (30_000, 0, 25), 1.5)


def test_ridge_between_ships_blocks_and_removal_clears(lgb, flat_dem):
    p1, p2 = (-8000, 0, 25), (8000, 0, 25)
    assert fresnel_clear(lgb, flat_dem, p1, p2, 1.5)
    flat_dem.set_box_enu(REF_LAT, REF_LON, -500, 500, -3000, 3000, 300.0)
    assert not fresnel_clear(lgb, flat_dem, p1, p2, 1.5)


def test_ridge_far_below_uav_path_does_not_block(lgb, flat_dem):
    flat_dem.set_box_enu(REF_LAT, REF_LON, -500, 500, -3000, 3000, 300.0)
    for f in (1.5, 16.0):
        assert fresnel_clear(lgb, flat_dem, (-8000, 0, 25), (8000, 0, 3000), f)


def test_ridge_above_uav_path_blocks(lgb, flat_dem):
    flat_dem.set_box_enu(REF_LAT, REF_LON, -500, 500, -3000, 3000, 3500.0)
    for f in (1.5, 16.0):
        assert not fresnel_clear(lgb, flat_dem, (-8000, 0, 25), (8000, 0, 3000), f)


def test_terrain_off_the_path_is_ignored(lgb, flat_dem):
    flat_dem.set_box_enu(REF_LAT, REF_LON, -2000, 2000, 5000, 9000, 500.0)   # 5 km north of path
    assert fresnel_clear(lgb, flat_dem, (-8000, 0, 25), (8000, 0, 25), 1.5)


def test_fresnel_result_is_symmetric(lgb, flat_dem):
    flat_dem.set_box_enu(REF_LAT, REF_LON, 2000, 3000, -3000, 3000, 200.0)
    a, b = (-8000, 0, 25), (8000, 500, 900)
    for f in (1.5, 16.0):
        assert fresnel_clear(lgb, flat_dem, a, b, f) == fresnel_clear(lgb, flat_dem, b, a, f)


def test_lower_frequency_has_larger_zone_and_blocks_first(lgb, flat_dem):
    """Plateau 12 m below a 1500 m level path: inside the 1.5 GHz 60% Fresnel zone
    (~17 m) but outside the 16 GHz one (~5 m). L must be blocked, Ku clear.
    Nodes are placed at 1500 m purely so the path is level over the plateau."""
    flat_dem.set_box_enu(REF_LAT, REF_LON, -7500, 7500, -3000, 3000, 1488.0)
    p1, p2 = (-8000, 0, 1500), (8000, 0, 1500)
    assert not fresnel_clear(lgb, flat_dem, p1, p2, lgb.FREQ_L_GHZ)
    assert fresnel_clear(lgb, flat_dem, p1, p2, lgb.FREQ_KU_GHZ)


# ------------------------------------------------------------------ LoS graphs
def _coords(*pts):
    return np.array(pts, dtype=float)


def test_los_graph_is_symmetric_with_zero_diagonal_and_3d_distances(lgb, flat_dem):
    c = _coords((0, 0, 25), (6000, 2000, 22), (-5000, 4000, 28), (1000, -7000, 20))
    adj = lgb.build_los_graph(c, flat_dem, REF_LAT, REF_LON).toarray()
    assert np.array_equal(adj > 0, (adj > 0).T)
    assert np.all(np.diag(adj) == 0)
    for i in range(4):
        for j in range(4):
            if adj[i, j] > 0:
                assert adj[i, j] == pytest.approx(np.linalg.norm(c[i] - c[j]))


def test_radio_horizon_gate(lgb, flat_dem):
    far = _coords((0, 0, 25), (45_000, 0, 25))           # horizon is ~35.7 km
    near = _coords((0, 0, 25), (12_000, 0, 25))
    assert lgb.build_los_graph(far, flat_dem, REF_LAT, REF_LON)[0, 1] == 0
    assert lgb.build_los_graph(near, flat_dem, REF_LAT, REF_LON)[0, 1] > 0


def test_dual_band_ship_ship_is_L_only(lgb, flat_dem):
    c = _coords((0, 0, 25), (8000, 0, 25), (0, 8000, 25), (3000, 3000, 2500))   # last = UAV
    g_l, g_ku = lgb.build_dual_band_graph(c, flat_dem, REF_LAT, REF_LON, n_ships=3)
    g_l, g_ku = g_l.toarray(), g_ku.toarray()
    assert g_l[:3, :3].max() > 0                 # ships hear each other on L
    assert g_ku[:3, :3].max() == 0               # no ship-ship Ku radio exists
    assert np.array_equal(g_l > 0, (g_l > 0).T) and np.array_equal(g_ku > 0, (g_ku > 0).T)


def test_dual_band_uav_links_exist_on_both_bands_over_open_sea(lgb, flat_dem):
    c = _coords((0, 0, 25), (8000, 0, 25), (3000, 3000, 2500))
    g_l, g_ku = lgb.build_dual_band_graph(c, flat_dem, REF_LAT, REF_LON, n_ships=2)
    assert (g_l.toarray()[2, :2] > 0).all() and (g_ku.toarray()[2, :2] > 0).all()


def test_dual_band_graph_can_have_ku_without_l(lgb, flat_dem):
    flat_dem.set_box_enu(REF_LAT, REF_LON, -7500, 7500, -3000, 3000, 1488.0)
    c = _coords((-8000, 0, 1500), (8000, 0, 1500))     # node 0 = "ship", node 1 = UAV
    g_l, g_ku = lgb.build_dual_band_graph(c, flat_dem, REF_LAT, REF_LON, n_ships=1)
    assert g_l[0, 1] == 0 and g_ku[0, 1] > 0


def test_ridge_removes_ship_ship_link_but_uav_link_survives(lgb, flat_dem):
    flat_dem.set_box_enu(REF_LAT, REF_LON, -500, 500, -6000, 6000, 300.0)
    c = _coords((-8000, 0, 25), (8000, 0, 25), (0, 0, 3000))
    g_l, g_ku = lgb.build_dual_band_graph(c, flat_dem, REF_LAT, REF_LON, n_ships=2)
    assert g_l[0, 1] == 0                              # ridge blocks ship-ship
    assert g_l[2, 0] > 0 and g_ku[2, 0] > 0            # UAV passes overhead


def test_dual_band_horizon_gate_applies_to_uav_links(lgb):
    """Horizon for a 25 m ship and a 1500 m UAV is 3.57*(sqrt(25)+sqrt(1500)) = ~156 km.
    A UAV at 100 km is reachable, one at 170 km is not, even over open sea.
    (place_uavs optimises with this function, so a wrong horizon here skews every placement.)"""
    from helpers import FakeDEM
    big = FakeDEM(REF_LAT, REF_LON, half_deg=2.5, res=0.01)           # +-277 km of open sea
    for dist_m, expect_link in ((100_000, True), (170_000, False)):
        c = _coords((0, 0, 25), (dist_m, 0, 1500))                    # node 1 = UAV
        g_l, g_ku = lgb.build_dual_band_graph(c, big, REF_LAT, REF_LON, n_ships=1)
        assert (g_l[0, 1] > 0 and g_ku[0, 1] > 0) == expect_link, f"{dist_m / 1000:.0f} km"
