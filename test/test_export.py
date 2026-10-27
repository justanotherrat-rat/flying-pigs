"""export_for_ns3 CSV contract, and consistency of what main() writes."""
import csv

import numpy as np
import pytest

from helpers import write_geotiff


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def make_history(n_ships=3, n_steps=2):
    ships = [np.array([[0, 0, 25.], [5000, 0, 25.], [0, 5000, 25.]][:n_ships]) + k * 100
             for k in range(n_steps)]
    uav = [np.array([2000., 2000., 1500. + k]) for k in range(n_steps)]
    n = n_ships + 1
    adj_l, adj_ku = [], []
    for _ in range(n_steps):
        l, ku = np.zeros((n, n)), np.zeros((n, n))
        l[0, 1] = l[1, 0] = 5000.0                       # ship-ship, L only
        l[0, n - 1] = l[n - 1, 0] = 2800.0               # ship-UAV on BOTH bands
        ku[0, n - 1] = ku[n - 1, 0] = 2800.0
        ku[1, n - 1] = ku[n - 1, 1] = 3500.0             # ship-UAV Ku only
        adj_l.append(l)
        adj_ku.append(ku)
    return ships, uav, adj_l, adj_ku


def test_nodes_csv_layout(lgb, tmp_path):
    ships, uav, al, ak = make_history()
    lgb.export_for_ns3(str(tmp_path), 300, ships, uav, al, ak)
    rows = read_csv(tmp_path / "nodes.csv")
    assert list(rows[0].keys()) == ["step", "t_s", "node", "x", "y", "z"]
    assert len(rows) == 2 * 4                              # steps * (ships + UAV)
    assert [int(r["node"]) for r in rows[:4]] == [0, 1, 2, 3]   # UAV is last
    assert {float(r["t_s"]) for r in rows} == {0.0, 300.0}
    uav_row = [r for r in rows if r["step"] == "1" and r["node"] == "3"][0]
    assert (float(uav_row["x"]), float(uav_row["z"])) == (2000.0, 1501.0)


def test_links_csv_has_one_row_per_band_and_upper_triangle_only(lgb, tmp_path):
    ships, uav, al, ak = make_history()
    lgb.export_for_ns3(str(tmp_path), 300, ships, uav, al, ak)
    rows = read_csv(tmp_path / "links.csv")
    assert list(rows[0].keys()) == ["step", "t_s", "i", "j", "band", "dist_m"]
    assert all(int(r["i"]) < int(r["j"]) for r in rows)
    step0 = [(int(r["i"]), int(r["j"]), r["band"]) for r in rows if r["step"] == "0"]
    assert sorted(step0) == [(0, 1, "L"), (0, 3, "Ku"), (0, 3, "L"), (1, 3, "Ku")]
    assert len(rows) == 2 * 4


def test_export_rejects_mismatched_histories(lgb, tmp_path):
    ships, uav, al, ak = make_history()
    with pytest.raises(AssertionError):
        lgb.export_for_ns3(str(tmp_path), 300, ships, uav[:1], al, ak)


def test_export_rejects_adjacency_without_uav_row(lgb, tmp_path):
    ships, uav, al, ak = make_history()
    al = [a[:-1, :-1] for a in al]
    with pytest.raises(AssertionError):
        lgb.export_for_ns3(str(tmp_path), 300, ships, uav, al, ak)


def test_export_warns_on_asymmetric_adjacency(lgb, tmp_path, capsys):
    ships, uav, al, ak = make_history()
    al[0][1, 0] = 0.0
    lgb.export_for_ns3(str(tmp_path), 300, ships, uav, al, ak)
    assert "not symmetric" in capsys.readouterr().out


# ------------------------------------------------- end-to-end through main()
@pytest.fixture
def main_run(lgb, monkeypatch, tmp_path):
    """Run main() on open sea with fixed ships; return parsed export."""
    tif = write_geotiff(tmp_path / "sea.tif", 1.4, 103.3, half_deg=0.8)
    fixed = np.array([[1.40, 103.20, 25.], [1.45, 103.25, 22.], [1.38, 103.30, 28.],
                      [1.42, 103.35, 24.]])
    monkeypatch.setattr(lgb, "generate_convoy_lla", lambda dem, n_ships=4, **k: fixed.copy())
    np.random.seed(0)
    lgb.main(sim_time=15, calc_interval=5, n_ships=4, placement_mode="midpoint",
             make_plots=False, dem_file_path=str(tif), export_dir=str(tmp_path / "exp"),
             plot_dir=str(tmp_path / "plots"))
    return (read_csv(tmp_path / "exp" / "nodes.csv"), read_csv(tmp_path / "exp" / "links.csv"))


def test_main_writes_expected_number_of_steps(main_run):
    nodes, _ = main_run
    assert {int(r["step"]) for r in nodes} == {0, 1, 2}
    assert len(nodes) == 3 * 5


def test_main_links_are_consistent_with_node_positions(main_run):
    nodes, links = main_run
    pos = {(int(r["step"]), int(r["node"])): np.array([float(r["x"]), float(r["y"]), float(r["z"])])
           for r in nodes}
    assert links, "expected at least one link on open sea"
    for r in links:
        s, i, j = int(r["step"]), int(r["i"]), int(r["j"])
        assert float(r["dist_m"]) == pytest.approx(np.linalg.norm(pos[s, i] - pos[s, j]), rel=1e-9)


def test_main_band_rules_hold(main_run):
    nodes, links = main_run
    uav = max(int(r["node"]) for r in nodes)
    for r in links:
        if r["band"] == "Ku":
            assert uav in (int(r["i"]), int(r["j"])), "Ku exists only on ship-UAV pairs"
        assert r["band"] in ("L", "Ku")


def test_main_ship_z_is_constant_and_uav_is_at_mid_altitude(main_run):
    nodes, _ = main_run
    ship_z = {n: {float(r["z"]) for r in nodes if int(r["node"]) == n} for n in range(4)}
    assert all(len(v) == 1 for v in ship_z.values())
    assert {float(r["z"]) for r in nodes if int(r["node"]) == 4} == {(1500.0 + 6100.0) / 2}   # midpoint mode


def test_main_ships_move_by_speed_times_interval(main_run):
    nodes, _ = main_run
    pos = {(int(r["step"]), int(r["node"])): np.array([float(r["x"]), float(r["y"])]) for r in nodes}
    for n in range(4):
        step = np.linalg.norm(pos[1, n] - pos[0, n])
        assert 9 * 0.514444 * 300 * 0.98 <= step <= 11 * 0.514444 * 300 * 1.02   # 9-11 kn, 5 min