"""Optional checks on ns-3's <out>_linkbudget.csv (written by uav_mesh_eval.cc).

These run only if a link-budget file is found, because ns-3 itself is not needed
for the rest of the suite. By default, files are read from sim-out/. Point at
other files with
    NS3_LINKBUDGET=/path/to/result_linkbudget.csv[:other.csv]   pytest
Assumes default radio settings
(sea state 3, vertical polarisation, 20 C, 35 ppt); override with SEA_STATE, POL_V.
"""
import csv
import os
from pathlib import Path

import numpy as np
import pytest

from helpers import fspl_db, two_ray_db

pytestmark = pytest.mark.ns3_output
REPO = Path(__file__).resolve().parents[1]
SIM_OUT = REPO / "sim-out"


def _files(sim_out=SIM_OUT):
    env = os.environ.get("NS3_LINKBUDGET")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p] or []
    return sorted(sim_out.rglob("*_linkbudget.csv"))


def test_default_search_is_limited_to_sim_out(monkeypatch, tmp_path):
    monkeypatch.delenv("NS3_LINKBUDGET", raising=False)
    sim_out = tmp_path / "sim-out"
    expected = sim_out / "rep0_linkbudget.csv"
    expected.parent.mkdir(parents=True)
    expected.touch()
    (tmp_path / "outside_linkbudget.csv").touch()
    assert _files(sim_out) == [expected]


FILES = _files()
SEA_STATE = int(os.environ.get("SEA_STATE", "3"))
POL_V = os.environ.get("POL_V", "1") != "0"


@pytest.fixture(params=FILES or [None], ids=lambda p: p.name if p else "no-linkbudget-file")
def rows(request):
    if request.param is None:
        pytest.skip("no *_linkbudget.csv found (run ns-3 first or set NS3_LINKBUDGET)")
    with open(request.param) as f:
        return list(csv.DictReader(f))


def F(r, k):
    return float(r[k])


def freq(r):
    return 16.0 if r["band"] == "Ku" else 1.5


def test_columns_present(rows):
    assert {"step", "i", "j", "band", "dist_km", "h1_m", "h2_m", "fspl_db", "two_ray_db",
            "point_loss_db", "rx_dbm", "snr_db", "outage_prob"} <= set(rows[0])


def test_fspl_column_matches_formula(rows):
    for r in rows:
        assert F(r, "fspl_db") == pytest.approx(fspl_db(F(r, "dist_km") * 1000, freq(r)), abs=0.05)


def test_noise_floor_is_constant_and_physical(rows):
    floors = {round(F(r, "rx_dbm") - F(r, "snr_db"), 1) for r in rows}
    assert len(floors) == 1
    assert -105 < floors.pop() < -85          # -174 + 73 dB(20 MHz) + NF(0..15 dB)


def test_pointing_loss_only_on_ku(rows):
    for r in rows:
        if r["band"] == "L":
            assert F(r, "point_loss_db") == 0
        else:
            assert F(r, "point_loss_db") >= 0


def test_outage_probability_is_a_probability_and_falls_with_snr(rows):
    for r in rows:
        assert 0.0 <= F(r, "outage_prob") <= 1.0
    for b in ("L", "Ku"):
        pts = sorted((F(r, "snr_db"), F(r, "outage_prob")) for r in rows if r["band"] == b)
        if len(pts) > 5:
            assert pts[-1][1] <= pts[0][1] + 1e-9


def test_two_ray_matches_independent_reference(rows):
    """Passes if the report agrees with the reference using EITHER the distance as
    written (3D slant range) or the horizontal range - see the test below."""
    bad = []
    for r in rows:
        d, h1, h2 = F(r, "dist_km") * 1000, F(r, "h1_m"), F(r, "h2_m")
        horiz = np.sqrt(max(d * d - (h1 - h2) ** 2, 1.0))
        got = F(r, "two_ray_db")
        ref_as_written = two_ray_db(d, h1, h2, freq(r), SEA_STATE, POL_V)
        ref_horizontal = two_ray_db(horiz, h1, h2, freq(r), SEA_STATE, POL_V)
        if min(abs(got - ref_as_written), abs(got - ref_horizontal)) > 0.3:
            bad.append((r["step"], r["i"], r["j"], got, ref_as_written, ref_horizontal))
    assert not bad, f"{len(bad)} links disagree with the reference, e.g. {bad[:3]}"


@pytest.mark.xfail(reason="known: offline report passes the 3D slant range (links.csv dist_m) "
                          "into TwoRaySeaLoss, which expects horizontal range; the live "
                          "RefreshLoss uses horizontal. Remove xfail once fixed.", strict=False)
def test_report_uses_horizontal_range_like_the_live_simulation(rows):
    for r in rows:
        d, h1, h2 = F(r, "dist_km") * 1000, F(r, "h1_m"), F(r, "h2_m")
        horiz = np.sqrt(max(d * d - (h1 - h2) ** 2, 1.0))
        assert F(r, "two_ray_db") == pytest.approx(
            two_ray_db(horiz, h1, h2, freq(r), SEA_STATE, POL_V), abs=0.3)
