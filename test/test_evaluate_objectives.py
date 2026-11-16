"""Tests for simulation/evaluate_objectives.py using tiny synthetic ns-3 outputs whose correct
verdicts are known by construction."""
import csv
import importlib.util
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("evaluate_objectives", REPO / "simulation" / "evaluate_objectives.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make_run(tmp, prefix="rep0", n_ships=3, steps=(0, 1), *, links=None, data=None, tel=None,
             lthr=None, etx=None, etx_rows=None, export=None):
    """Write nodes.csv + the ns-3 CSVs. links: [(step,i,j,band,rx_dbm,outage)]; data/tel: per-ship dicts."""
    export = Path(export or tmp / f"export_{prefix}")
    export.mkdir(parents=True, exist_ok=True)
    uav = n_ships
    _w(export / "nodes.csv", ["step", "t_s", "node", "x", "y", "z"],
       [(s, s * 300, n, n * 1000, 0, 25 if n < n_ships else 1500) for s in steps for n in range(n_ships + 1)])
    if links is None:        # every ship hears the UAV on both bands, and ship0-ship1 on L
        links = [(s, i, uav, b, -60.0, 0.01) for s in steps for i in range(n_ships) for b in ("L", "Ku")]
        links += [(s, 0, 1, "L", -70.0, 0.01) for s in steps]
    _w(tmp / f"{prefix}_linkbudget.csv",
       ["step", "i", "j", "band", "dist_km", "h1_m", "h2_m", "fspl_db", "two_ray_db", "point_loss_db",
        "rx_dbm", "snr_db", "outage_prob"],
       [(s, i, j, b, 1, 25, 1500, 0, 0, 0, rx, 40, out) for s, i, j, b, rx, out in links])
    data = data or {s: dict(offered=1000, rx=1000, delay=1.5, kbps=11.0) for s in range(n_ships)}
    _w(tmp / f"{prefix}_datatransfer.csv",
       ["ship", "offered_pkts", "sent_pkts", "noroute_pkts", "rx_pkts", "pdr", "mean_delay_ms", "mean_hops",
        "throughput_kbps"],
       [(s, d["offered"], d["offered"], 0, d["rx"], d["rx"] / d["offered"], d["delay"], 1.0, d["kbps"])
        for s, d in data.items()])
    tel = tel or {s: dict(rx=10, delay=0.6) for s in range(n_ships)}
    _w(tmp / f"{prefix}_telemetry.csv",
       ["ship", "possible_receptions", "sent_pkts", "rx_receptions", "pdr", "mean_delay_ms", "mean_hops"],
       [(s, 10, 10, t["rx"], t["rx"] / 10, t["delay"], 1.0) for s, t in tel.items()])
    if lthr is not None:
        _w(tmp / f"{prefix}_lthroughput.csv",
           ["ship", "offered_pkts", "sent_pkts", "noroute_pkts", "rx_pkts", "pdr", "mean_delay_ms", "mean_hops",
            "throughput_kbps"],
           [(s, d["offered"], d["offered"], 0, d["rx"], d["rx"] / d["offered"], 2.0, 1.0, d["kbps"])
            for s, d in lthr.items()])
    if etx is not None:
        _w(tmp / f"{prefix}_etx.csv", ["src", "dst", "class", "hops", "samples", "no_route_samples",
                                       "unmeasured_samples", "route_etx"],
           [(s, uav, "data", 1, 5, 0, 0, v) for s, v in etx.items()])
    if etx_rows is not None:
        _w(tmp / f"{prefix}_etx.csv", ["src", "dst", "class", "hops", "samples", "no_route_samples",
                                       "unmeasured_samples", "route_etx"], etx_rows)
    return export


def score(ev, tmp, obj=None, **kw):
    export = make_run(tmp, **kw)
    run = ev.Run(0, "run", kw.get("prefix", "rep0"), tmp, export)
    return ev.score_run(run, obj or ev.Objectives())


# ------------------------------------------------------------------ comparison strictness
def test_signal_threshold_is_inclusive(ev, tmp_path):
    links = [(s, i, 3, "Ku", -100.0, 0.0) for s in (0, 1) for i in range(3)]
    assert score(ev, tmp_path, links=links).verdict["signal"] == ev.PASS
    links = [(s, i, 3, "Ku", -100.01, 0.0) for s in (0, 1) for i in range(3)]
    assert score(ev, tmp_path, links=links, prefix="b").verdict["signal"] == ev.FAIL


def test_latency_is_strictly_below_50ms(ev, tmp_path):
    ok = {s: dict(offered=100, rx=100, delay=49.99, kbps=1) for s in range(3)}
    at = {s: dict(offered=100, rx=100, delay=50.0, kbps=1) for s in range(3)}
    assert score(ev, tmp_path, data=ok).verdict["latency"] == ev.PASS
    assert score(ev, tmp_path, data=at, prefix="b").verdict["latency"] == ev.FAIL


def test_latency_counts_telemetry_too(ev, tmp_path):
    slow_tel = {0: dict(rx=10, delay=80.0), 1: dict(rx=10, delay=1.0), 2: dict(rx=10, delay=1.0)}
    assert score(ev, tmp_path, tel=slow_tel).verdict["latency"] == ev.FAIL


def test_undelivered_flow_fails_latency_instead_of_passing_on_zero(ev, tmp_path):
    data = {s: dict(offered=100, rx=(0 if s == 1 else 100), delay=0.0, kbps=1) for s in range(3)}
    sc = score(ev, tmp_path, data=data)
    assert sc.ships[1].verdict["latency"] == ev.FAIL and math.isinf(sc.ships[1].data_latency_ms)


def test_etx_strictly_below_2_2_modeled_and_measured_override(ev, tmp_path):
    ok = [(s, i, 3, "Ku", -60, 0.5) for s in (0, 1) for i in range(3)]          # ETX 2.0
    bad = [(s, i, 3, "Ku", -60, 0.55) for s in (0, 1) for i in range(3)]        # ETX 2.22
    a, b = score(ev, tmp_path, links=ok), score(ev, tmp_path, links=bad, prefix="b")
    assert a.verdict["etx"] == ev.PASS and b.verdict["etx"] == ev.FAIL
    assert a.ships[0].etx == pytest.approx(2.0) and a.ships[0].etx_source == "modeled"
    assert any("MODELED" in n for n in a.notes)
    m = score(ev, tmp_path, links=bad, prefix="c", etx={0: 1.1, 1: 1.3, 2: 2.2})  # measured wins; 2.2 fails
    assert m.ships[0].etx_source == "measured" and m.ships[2].verdict["etx"] == ev.FAIL
    assert m.ships[0].verdict["etx"] == ev.PASS and not any("MODELED" in n for n in m.notes)


def test_etx_step_without_a_ku_link_means_no_route(ev, tmp_path):
    links = [(0, i, 3, "Ku", -60, 0.0) for i in range(3)] + [(1, i, 3, "L", -60, 0.0) for i in range(3)]
    sc = score(ev, tmp_path, links=links)
    assert math.isinf(sc.ships[0].etx) and sc.verdict["etx"] == ev.FAIL


def test_measured_etx_uses_the_worst_route_per_ship_and_ignores_nan_rows(ev, tmp_path):
    rows = [(0, 1, "telemetry", 1, 5, 0, 0, 1.1), (0, 2, "telemetry", 2, 5, 0, 0, 2.5),      # worst 2.5 -> FAIL
            (1, 2, "telemetry", 1, 5, 0, 0, 1.2), (1, 3, "data", 1, 5, 0, 0, 1.0),          # PASS
            (2, 0, "telemetry", 1, 0, 0, 5, float("nan")), (2, 1, "telemetry", 1, 5, 0, 0, 1.4)]
    sc = score(ev, tmp_path, etx_rows=rows)
    assert sc.ships[0].etx == 2.5 and sc.ships[0].verdict["etx"] == ev.FAIL
    assert sc.ships[1].etx == 1.2 and sc.ships[1].verdict["etx"] == ev.PASS
    assert sc.ships[2].etx == 1.4                                    # the NaN row did not poison it
    mean = score(ev, tmp_path, ev.Objectives(etx_stat="mean"), etx_rows=rows, prefix="b")
    assert mean.ships[0].etx == pytest.approx(1.8)


def test_unmeasured_only_ship_is_not_measurable_and_route_less_ship_fails(ev, tmp_path):
    nan, inf = float("nan"), float("inf")
    rows = [(0, 1, "telemetry", 1, 5, 0, 0, 1.1),
            (1, 0, "telemetry", 1, 0, 0, 5, nan),                  # ship 1: only unmeasured routes
            (2, 0, "telemetry", 0, 0, 5, 0, inf)]                  # ship 2: no route exists at all
    sc = score(ev, tmp_path, etx_rows=rows)
    assert sc.ships[1].verdict["etx"] == ev.NM and sc.verdict["etx"] == ev.NM
    assert sc.ships[2].verdict["etx"] == ev.FAIL and math.isinf(sc.ships[2].etx)
    assert sc.ships[0].verdict["etx"] == ev.PASS


def test_ship_missing_from_the_etx_file_entirely_has_no_route(ev, tmp_path):
    sc = score(ev, tmp_path, etx_rows=[(0, 1, "telemetry", 1, 5, 0, 0, 1.1)])
    assert math.isinf(sc.ships[1].etx) and sc.ships[1].verdict["etx"] == ev.FAIL


# ------------------------------------------------------------------ signal: band handling
def test_signal_takes_the_worst_band_not_the_last_row_written(ev, tmp_path):
    """Regression guard: run_replications.route_link_metrics keeps only ONE row per (ship, step),
    so a good Ku row written after a bad L row hides the bad L link. Here the L link is -105 dBm."""
    links = []
    for s in (0, 1):
        for i in range(3):
            links += [(s, i, 3, "L", -105.0, 0.0), (s, i, 3, "Ku", -60.0, 0.0)]    # Ku last
    sc = score(ev, tmp_path, links=links)
    assert sc.verdict["signal"] == ev.FAIL and sc.ships[0].signal_dbm == -105.0


def test_scope_uav_ignores_ship_ship_links_scope_all_counts_them(ev, tmp_path):
    links = [(s, i, 3, "Ku", -60.0, 0.0) for s in (0, 1) for i in range(3)]
    links += [(s, 0, 1, "L", -110.0, 0.0) for s in (0, 1)]                          # bad ship-ship link
    assert score(ev, tmp_path, links=links).ships[0].verdict["signal"] == ev.FAIL      # all (default)
    uav = score(ev, tmp_path, ev.Objectives(signal_scope="uav"), links=links, prefix="b")
    assert uav.verdict["signal"] == ev.PASS


def test_isolated_ship_fails_signal(ev, tmp_path):
    links = [(0, i, 3, "Ku", -60.0, 0.0) for i in range(3)] + [(1, i, 3, "Ku", -60.0, 0.0) for i in (0, 1)]
    sc = score(ev, tmp_path, links=links)                       # ship 2 has no link at step 1
    assert sc.ships[2].verdict["signal"] == ev.FAIL and sc.ships[2].isolated_steps == 1
    assert sc.ships[2].signal_dbm == -60.0                      # its remaining link was fine; isolation is the failure


# ------------------------------------------------------------------ throughput: three outcomes
def test_current_trickle_traffic_is_not_measurable_not_a_fail(ev, tmp_path):
    sc = score(ev, tmp_path)                                    # 11 kbps offered per ship
    assert sc.verdict["ku_throughput"] == ev.NM and sc.verdict["l_throughput"] == ev.NM
    assert any("offered load" in n for n in sc.notes)


def test_ku_per_ship_and_aggregate_targets_scale_with_ship_count(ev):
    o = ev.Objectives()
    assert o.ku_per_ship_mbps == pytest.approx(1000 / 30)
    assert o.ku_total_target(5) == pytest.approx(166.667, abs=1e-3)


def test_ku_saturated_pass_and_fail_use_the_aggregate(ev, tmp_path):
    kb = 40_000.0                                               # 40 Mbps per ship offered and delivered
    good = {s: dict(offered=1000, rx=1000, delay=2.0, kbps=kb) for s in range(3)}
    assert score(ev, tmp_path, data=good).verdict["ku_throughput"] == ev.PASS          # 120 > 100
    short = {s: dict(offered=1000, rx=500, delay=2.0, kbps=kb / 2) for s in range(3)}  # 60 total < 100
    sc = score(ev, tmp_path, data=short, prefix="b")
    assert sc.verdict["ku_throughput"] == ev.FAIL and sc.ku_total_mbps == pytest.approx(60.0)


def test_ku_throughput_exactly_at_target_is_not_a_pass(ev, tmp_path):
    exact = 1000 / 30 * 1000                                    # kbps
    data = {s: dict(offered=1000, rx=1000, delay=2.0, kbps=exact) for s in range(3)}
    assert score(ev, tmp_path, data=data).verdict["ku_throughput"] == ev.FAIL          # '>' not '>='


def test_per_ship_throughput_exactly_at_target_fails_for_both_bands(ev, tmp_path):
    """Round numbers so float rounding cannot hide an off-by-one in the '>' rule.
    Ku: 90 Mbps over 3 nodes = exactly 30.0 per ship.  L: exactly 5.0 per ship."""
    obj = ev.Objectives(ku_total_mbps_ref=90.0, ku_ref_nodes=3)
    ku = {s: dict(offered=1000, rx=1000, delay=2.0, kbps=30_000.0) for s in range(3)}
    l = {s: dict(offered=1000, rx=1000, kbps=5_000.0) for s in range(3)}
    sc = score(ev, tmp_path, obj, data=ku, lthr=l)
    assert all(x.verdict["ku_throughput"] == ev.FAIL for x in sc.ships)
    assert all(x.verdict["l_throughput"] == ev.FAIL for x in sc.ships)
    assert sc.verdict["ku_throughput"] == ev.FAIL and sc.verdict["l_throughput"] == ev.FAIL
    just_above = {s: dict(offered=1000, rx=1000, kbps=5_000.1) for s in range(3)}
    assert score(ev, tmp_path, obj, lthr=just_above, prefix="b").verdict["l_throughput"] == ev.PASS


def test_known_offered_rate_lets_a_zero_delivery_run_fail_instead_of_nm(ev, tmp_path):
    data = {s: dict(offered=1000, rx=0, delay=0.0, kbps=0.0) for s in range(3)}
    assert score(ev, tmp_path, data=data, prefix="a").verdict["ku_throughput"] == ev.NM
    obj = ev.Objectives(ku_offered_mbps=50.0)
    assert score(ev, tmp_path, obj, data=data, prefix="b").verdict["ku_throughput"] == ev.FAIL


def test_l_band_throughput_from_lthroughput_csv(ev, tmp_path):
    good = {s: dict(offered=1000, rx=1000, kbps=6000.0) for s in range(3)}
    assert score(ev, tmp_path, lthr=good).verdict["l_throughput"] == ev.PASS
    sc = score(ev, tmp_path, lthr={0: dict(offered=1000, rx=1000, kbps=6000.0),
                                   1: dict(offered=1000, rx=400, kbps=2400.0),
                                   2: dict(offered=1000, rx=1000, kbps=6000.0)}, prefix="b")
    assert sc.verdict["l_throughput"] == ev.FAIL and sc.ships[1].verdict["l_throughput"] == ev.FAIL


# ------------------------------------------------------------------ statistics
def test_wilson_interval_known_values(ev):
    lo, hi = ev.wilson(8, 10)
    assert (lo, hi) == pytest.approx((0.4902, 0.9433), abs=1e-3)
    assert ev.wilson(0, 0)[0] != ev.wilson(0, 0)[0]              # nan
    lo, hi = ev.wilson(10, 10)
    assert hi == 1.0 and lo > 0.69


def test_mean_ci_matches_textbook(ev):
    m, hw = ev.mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m == 3.0 and hw == pytest.approx(2.776 * (2.5 ** 0.5) / (5 ** 0.5), rel=1e-3)
    assert math.isnan(ev.mean_ci([1.0])[1])


# ------------------------------------------------------------------ discovery and pairing
def _compare_layout(ev, tmp_path, opt_rx, mid_rx, reps=4):
    work = tmp_path / "work"
    for i in range(reps):
        for placement, rx in (("optimized", opt_rx), ("midpoint", mid_rx)):
            export = work / "exports" / f"rep{i}" / placement
            links = [(s, j, 3, "Ku", rx[i], 0.0) for s in (0, 1) for j in range(3)]
            make_run(tmp_path, f"rep{i}_{placement}", links=links, export=export)
    return work


def test_discovery_handles_both_layouts(ev, tmp_path):
    work = _compare_layout(ev, tmp_path, [-60] * 4, [-60] * 4, reps=2)
    runs = ev.discover_runs(work, "rep", tmp_path)
    assert [(r.replication, r.placement, r.prefix) for r in runs] == [
        (0, "midpoint", "rep0_midpoint"), (0, "optimized", "rep0_optimized"),
        (1, "midpoint", "rep1_midpoint"), (1, "optimized", "rep1_optimized")]
    single = tmp_path / "single"
    make_run(tmp_path, "rep7", export=single / "exports" / "rep7")
    r = ev.discover_runs(single, "rep", tmp_path, "optimized")
    assert [(x.replication, x.placement, x.prefix) for x in r] == [(7, "optimized", "rep7")]


def test_paired_delta_sign_and_wins(ev, tmp_path):
    # optimized keeps every link above -100 dBm in all 4 scenarios; midpoint fails in 3 of 4
    work = _compare_layout(ev, tmp_path, [-70, -70, -70, -70], [-70, -105, -105, -105])
    scores = [ev.score_run(r, ev.Objectives()) for r in ev.discover_runs(work, "rep", tmp_path)]
    pr, npairs = ev.paired(scores)
    assert npairs == 4
    sig = pr["signal"]
    assert sig["rate_delta"][0] == pytest.approx(0.75) and (sig["wins"], sig["losses"], sig["ties"]) == (3, 0, 1)
    assert sig["scalar_delta"][0] > 0                          # worst received power is HIGHER for optimized
    assert pr["ku_throughput"]["n"] == 0                       # not measurable -> excluded, not zero


def test_end_to_end_cli_report_csv_and_markdown(ev, tmp_path, capsys):
    work = _compare_layout(ev, tmp_path, [-70] * 4, [-105] * 4)
    ev.main(["--ns3-root", str(tmp_path), "--work-dir", str(work), "--csv-out", str(tmp_path / "o.csv"),
             "--md-out", str(tmp_path / "o.md")])
    out = capsys.readouterr().out
    assert "== optimized: 4 run(s) ==" in out and "== midpoint: 4 run(s) ==" in out
    assert "PAIRED  optimized - midpoint" in out and "n/a (not measurable)" in out
    rows = list(csv.DictReader(open(tmp_path / "o.csv")))
    assert len(rows) == 4 * 2 * 3 and {r["signal_verdict"] for r in rows} == {"PASS", "FAIL"}
    assert "| Received power |" in (tmp_path / "o.md").read_text()


def test_missing_files_are_reported_not_crashed(ev, tmp_path, capsys):
    work = _compare_layout(ev, tmp_path, [-70] * 2, [-70] * 2, reps=2)
    (tmp_path / "rep1_midpoint_datatransfer.csv").unlink()
    ev.main(["--ns3-root", str(tmp_path), "--work-dir", str(work)])
    assert "SKIPPED" in capsys.readouterr().out
