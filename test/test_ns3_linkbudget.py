"""Optional checks on ns-3's <out>_linkbudget.csv (written by uav_mesh_eval.cc).

These run only if a link-budget file is found, because ns-3 itself is not needed
for the rest of the suite. Point at files with
    NS3_LINKBUDGET=/path/to/result_linkbudget.csv[:other.csv]   pytest
or drop *_linkbudget.csv anywhere in the repo. Assumes default radio settings
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


def _files():
    env = os.environ.get("NS3_LINKBUDGET")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p] or []
    return sorted(REPO.rglob("*_linkbudget.csv"))


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


# The C++ writes dist_km and heights with 2 decimals and losses with 2 decimals, so a
# value recomputed from the CSV can legitimately differ from the simulator's own
# (unrounded) number. Tolerances below are derived from that precision, not guessed.
DIST_ROUND_M = 5.0      # dist_km rounded to 0.01 km  -> +-5 m
H_ROUND_M = 0.005       # heights rounded to 0.01 m   -> +-0.005 m
OUT_ROUND_DB = 0.01     # losses written with 2 decimals


def test_fspl_column_matches_formula(rows):
    for r in rows:
        d = F(r, "dist_km") * 1000
        tol = 8.686 * DIST_ROUND_M / max(d, 1.0) + OUT_ROUND_DB      # d(FSPL) = 8.686 * dd/d
        assert F(r, "fspl_db") == pytest.approx(fspl_db(d, freq(r)), abs=tol), (
            f"step {r['step']} link {r['i']}-{r['j']} {r['band']}: d={d:.0f} m")


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


def _envelope(d3, h1, h2, f, horizontal):
    """Min/max reference loss over every (d, h1, h2) that rounds to the CSV values."""
    vals = []
    for dd in np.linspace(-DIST_ROUND_M, DIST_ROUND_M, 21):
        for e1 in (-H_ROUND_M, H_ROUND_M):
            for e2 in (-H_ROUND_M, H_ROUND_M):
                a, b, dist = h1 + e1, h2 + e2, d3 + dd
                d = np.sqrt(max(dist * dist - (a - b) ** 2, 1.0)) if horizontal else dist
                vals.append(two_ray_db(d, a, b, f, SEA_STATE, POL_V))
    return min(vals) - OUT_ROUND_DB, max(vals) + OUT_ROUND_DB


def test_two_ray_matches_independent_reference(rows):
    """Passes if the reported loss lies inside the reference's range over the CSV's rounding
    interval, using EITHER the distance as written (3D slant range) or the horizontal range
    (see the xfail test below). A real error has to exceed the rounding envelope to fail."""
    bad = []
    for r in rows:
        d, h1, h2, got = F(r, "dist_km") * 1000, F(r, "h1_m"), F(r, "h2_m"), F(r, "two_ray_db")
        ok = False
        for horizontal in (False, True):
            lo, hi = _envelope(d, h1, h2, freq(r), horizontal)
            ok = ok or (lo <= got <= hi)
        if not ok:
            lo, hi = _envelope(d, h1, h2, freq(r), False)
            bad.append((r["step"], r["i"], r["j"], r["band"], got, round(lo, 2), round(hi, 2)))
    assert not bad, f"{len(bad)} links outside the reference envelope, e.g. {bad[:3]}"

def test_report_uses_horizontal_range_like_the_live_simulation(rows):
    """Only links where the two conventions give distinguishable answers (rounding envelopes
    do not overlap) can decide this; typically ship-UAV links with a large height difference."""
    decided = 0
    for r in rows:
        d, h1, h2, got = F(r, "dist_km") * 1000, F(r, "h1_m"), F(r, "h2_m"), F(r, "two_ray_db")
        lo_s, hi_s = _envelope(d, h1, h2, freq(r), False)
        lo_h, hi_h = _envelope(d, h1, h2, freq(r), True)
        if hi_h < lo_s or hi_s < lo_h:
            decided += 1
            assert lo_h <= got <= hi_h, f"step {r['step']} link {r['i']}-{r['j']} {r['band']}"
    if not decided:
        pytest.skip("no link in this file distinguishes slant from horizontal range")