"""Validation of the ns-3 maritime physics against Recommendation ITU-R P.527-6.

WHAT THIS FILE DOES
-------------------
uav_mesh_eval.cc implements sea-water permittivity (P.527-6 Sec. 5.1.2, eqs 14-27) and a
Fresnel reflection coefficient (P.527-6 Sec. 6, eqs 91-92).  This file compiles THOSE EXACT
C++ FUNCTIONS (extracted from the .cc at test time, no ns-3 needed) and checks them against
an implementation written directly from the Recommendation's equations in 50-digit
arithmetic (mpmath).  Because the C++ text is re-extracted on every run, editing the
physics code and forgetting to update anything makes this file fail.

WHAT P.527-6 DOES AND DOES NOT PUBLISH  (read before citing these tests)
--------------------------------------------------------------------------
* P.527-6 contains NO numerical worked examples.  It gives equations (1)-(99) and graphs
  (Fig. 2: pure/sea water at 20 C; Fig. 3: at 0 C; Fig. 21: sea-water emissivity at
  20 C / 35 ppt).  There is therefore nothing in the text to "reproduce exactly" as a
  tabulated answer.
* What the Recommendation DOES state, and what is tested here:
    - the full equation set for sea water (eqs 5-27) and the Fresnel equations (91-96);
    - S = 0 reduces sea water to pure water (text after eq 27);
    - valid range -4 <= T <= 40 C, 0 <= S <= 40 ppt (Sec. 5.1);
    - sigma = 2*pi*eps0*f*eps'' ~ 0.05563*f_GHz*eps'' (eq 3a) -> the "18*sigma/f" term;
    - at normal incidence r_v = -r_h and r_c = 0 (eqs 94-96).
* Not covered by P.527-6 and therefore NOT validated against it here: the two-ray
  interference sum, the Ament roughness factor, the sea-state -> RMS-height table.
  Those are only checked for self-consistency with the Python mirror in helpers.py (class
  MirrorIntegrity below), which test_ns3_linkbudget.py uses as its oracle.
* The "regression pins" below were computed from the Recommendation's equations with
  mpmath; they are NOT numbers printed by ITU.  They exist so that a change that moves the
  C++ and the in-file reference together is still caught.

Environment
-----------
  ITU_P527_CC=/path/to/uav_mesh_eval.cc   override the source file
  ITU_P527_REQUIRE_CPP=1                  fail (not skip) if no C++ compiler is found
                                          (also implied when CI is set)
Needs: g++ (or $CXX), numpy, mpmath.
"""
import hashlib
import itertools
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from mpmath import mp, mpf, mpc, sqrt as mpsqrt, exp as mpexp, cos as mpcos, sin as mpsin

from helpers import sea_permittivity as mirror_sea_permittivity, two_ray_db as mirror_two_ray_db

mp.dps = 50
REPO = Path(__file__).resolve().parents[1]
CC_SOURCE = Path(os.environ.get("ITU_P527_CC", REPO / "simulation" / "uav_mesh_eval.cc"))
BEGIN_MARK = "MARITIME (TWO-RAY SEA-REFLECTION) PATH LOSS MODEL"
END_MARK = "CAVEATS - what the two-ray model above does NOT capture"


# =========================================================================
# 1. Reference implementation, written from the Recommendation (mpmath)
#    Variable names follow the Rec.; equation numbers are P.527-6's.
# =========================================================================
def _water_terms(T):
    T = mpf(T)
    theta = 300 / (T + mpf("273.15")) - 1                       # (11)
    eps_s = mpf("77.66") + mpf("103.3") * theta                 # (8)
    eps_1 = mpf("0.0671") * eps_s                               # (9)
    eps_inf = mpf("3.52") - mpf("7.52") * theta                 # (10)
    f1 = mpf("20.20") - mpf("146.4") * theta + 316 * theta**2   # (12)
    f2 = mpf("39.8") * f1                                       # (13)
    return eps_s, eps_1, eps_inf, f1, f2


def ref_pure_water(f_ghz, T):
    """Eqs (5)-(7): pure water, returned as (eps', eps'')."""
    f = mpf(f_ghz)
    eps_s, eps_1, eps_inf, f1, f2 = _water_terms(T)
    re = (eps_s - eps_1) / (1 + (f / f1) ** 2) + (eps_1 - eps_inf) / (1 + (f / f2) ** 2) + eps_inf
    im = (f / f1) * (eps_s - eps_1) / (1 + (f / f1) ** 2) \
        + (f / f2) * (eps_1 - eps_inf) / (1 + (f / f2) ** 2)
    return re, im


def ref_sigma_sw(T, S):
    """Eqs (22)-(27): sea-water ionic conductivity in S/m."""
    T, S = mpf(T), mpf(S)
    sigma35 = (mpf("2.903602") + mpf("8.607e-2") * T + mpf("4.738817e-4") * T**2
               - mpf("2.991e-6") * T**3 + mpf("4.3047e-9") * T**4)                    # (23)
    R15 = S * (mpf("37.5109") + mpf("5.45216") * S + mpf("1.4409e-2") * S**2) \
        / (mpf("1004.75") + mpf("182.283") * S + S**2)                                # (24)
    alpha0 = (mpf("6.9431") + mpf("3.2841") * S - mpf("9.9486e-2") * S**2) \
        / (mpf("84.850") + mpf("69.024") * S + S**2)                                  # (26)
    alpha1 = mpf("49.843") - mpf("0.2276") * S + mpf("0.198e-2") * S**2              # (27)
    RT15 = 1 + alpha0 * (T - 15) / (alpha1 + T)                                       # (25)
    return sigma35 * R15 * RT15                                                       # (22)


def ref_sea_water(f_ghz, T, S):
    """Eqs (14)-(21): sea water, returned as (eps', eps'') with eps_sw = eps' - j eps''."""
    f, T, S = mpf(f_ghz), mpf(T), mpf(S)
    eps_s, eps_1, eps_inf, f1, f2 = _water_terms(T)
    eps_ss = eps_s * mpexp(mpf("-3.33330e-3") * S + mpf("4.74868e-6") * S**2)               # (17)
    f1s = f1 * (1 + S * (mpf("2.3232e-3") - mpf("7.9208e-5") * T + mpf("3.6764e-6") * T**2
                         + mpf("3.5594e-7") * T**3 + mpf("8.9795e-9") * T**4))              # (18)
    eps_1s = eps_1 * mpexp(mpf("-6.28908e-3") * S + mpf("1.76032e-4") * S**2
                           - mpf("9.22144e-5") * T * S)                                      # (19)
    f2s = f2 * (1 + S * (mpf("-1.99723e-2") + mpf("1.81176e-4") * T))                        # (20)
    eps_infs = eps_inf * (1 + S * (mpf("-2.04265e-3") + mpf("1.57883e-4") * T))              # (21)
    sig = ref_sigma_sw(T, S)
    re = ((eps_ss - eps_1s) / (1 + (f / f1s) ** 2)
          + (eps_1s - eps_infs) / (1 + (f / f2s) ** 2) + eps_infs)                           # (15)
    im = ((f / f1s) * (eps_ss - eps_1s) / (1 + (f / f1s) ** 2)
          + (f / f2s) * (eps_1s - eps_infs) / (1 + (f / f2s) ** 2) + 18 * sig / f)           # (16)
    return re, im


def ref_fresnel_grazing(psi_rad, eps_re, eps_im, vertical):
    """Eqs (91)/(92) with theta = angle from the normal = pi/2 - grazing angle psi."""
    theta = mp.pi / 2 - mpf(psi_rad)
    eps = mpc(eps_re, -eps_im)                                   # eps_r' - j eps_r'' (eq 1b)
    root = mpsqrt(eps - mpsin(theta) ** 2)
    c = mpcos(theta)
    return (eps * c - root) / (eps * c + root) if vertical else (c - root) / (c + root)


# =========================================================================
# 2. The real C++ functions, compiled from simulation/uav_mesh_eval.cc
# =========================================================================
HARNESS_PRE = r"""
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <string>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static double g_seaTempC = 20.0;
static double g_salinityPpt = 35.0;
"""
HARNESS_POST = r"""
int main()
{
    std::string cmd;
    while (std::cin >> cmd)
    {
        if (cmd == "eps")
        {
            double f, T, S;
            std::cin >> f >> T >> S;
            SeaPermittivity k = SeaWaterPermittivity(f, T, S);
            std::printf("%.17g %.17g\n", k.epsRe, k.epsIm);
        }
        else if (cmd == "refl")
        {
            double psi, f, T, S;
            int v;
            std::cin >> psi >> f >> v >> T >> S;
            g_seaTempC = T;
            g_salinityPpt = S;
            std::complex<double> r = FresnelReflectionCoefficient(psi, f, v != 0);
            std::printf("%.17g %.17g\n", r.real(), r.imag());
        }
        else if (cmd == "tworay")
        {
            double d, h1, h2, f, T, S;
            int ss, v;
            std::cin >> d >> h1 >> h2 >> f >> ss >> v >> T >> S;
            g_seaTempC = T;
            g_salinityPpt = S;
            std::printf("%.17g\n", TwoRaySeaLoss(d, h1, h2, f, ss, v != 0));
        }
    }
    return 0;
}
"""


def _require(reason):
    if os.environ.get("CI") or os.environ.get("ITU_P527_REQUIRE_CPP") == "1":
        pytest.fail(reason)
    pytest.skip(reason + "  (set ITU_P527_REQUIRE_CPP=1 to make this a failure)")


class CppPhysics:
    def __init__(self, exe):
        self.exe = exe

    def _run(self, lines):
        out = subprocess.run([str(self.exe)], input="\n".join(lines) + "\n", capture_output=True,
                             text=True, timeout=120, check=True).stdout.split("\n")
        return [tuple(float(x) for x in ln.split()) for ln in out if ln.strip()]

    def eps(self, cases):          # [(f_ghz, T, S)] -> [(eps', eps'')]
        return self._run([f"eps {f!r} {T!r} {S!r}" for f, T, S in cases])

    def refl(self, cases):         # [(psi_rad, f_ghz, vertical, T, S)] -> [(re, im)]
        return self._run([f"refl {p!r} {f!r} {int(v)} {T!r} {S!r}" for p, f, v, T, S in cases])

    def tworay(self, cases):       # [(d,h1,h2,f,seastate,vertical,T,S)] -> [(loss_db,)]
        return self._run([f"tworay {d!r} {h1!r} {h2!r} {f!r} {ss} {int(v)} {T!r} {S!r}"
                          for d, h1, h2, f, ss, v, T, S in cases])


@pytest.fixture(scope="session")
def cpp(tmp_path_factory):
    if not CC_SOURCE.is_file():
        pytest.fail(f"physics source not found: {CC_SOURCE}")
    lines = CC_SOURCE.read_text().splitlines()
    try:
        b = next(i for i, ln in enumerate(lines) if BEGIN_MARK in ln)
        e = next(i for i, ln in enumerate(lines) if END_MARK in ln)
    except StopIteration:
        pytest.fail(f"could not find the section markers {BEGIN_MARK!r} / {END_MARK!r} in "
                    f"{CC_SOURCE}; if you renamed them, update BEGIN_MARK/END_MARK here")
    assert b < e
    block = "\n".join(lines[b - 1:e])    # b-1 keeps the '// ====' banner line above the marker
    src = HARNESS_PRE + block + HARNESS_POST
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if not cxx:
        _require("no C++ compiler found")
    d = tmp_path_factory.mktemp("p527_cpp")
    tag = hashlib.sha1(src.encode()).hexdigest()[:10]
    cc, exe = d / f"p527_{tag}.cc", d / f"p527_{tag}"
    cc.write_text(src)
    r = subprocess.run([cxx, "-std=c++17", "-O1", "-Wno-unused-function", "-o", str(exe), str(cc)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail("the extracted physics block no longer compiles standalone (does it now "
                    f"depend on ns-3 or a new global?):\n{r.stderr[:3000]}")
    return CppPhysics(exe)


def rel_err(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# =========================================================================
# 3. Test grids
# =========================================================================
FREQS = [0.1, 0.5, 1.5, 5.0, 10.0, 16.0, 30.0, 60.0, 100.0, 300.0, 1000.0]   # GHz; Rec: up to 1000
TEMPS = [-4.0, 0.0, 10.0, 20.0, 30.0, 40.0]                                    # Rec valid range
SALS = [0.0, 5.0, 20.0, 35.0, 40.0]                                            # Rec valid range
GRID = list(itertools.product(FREQS, TEMPS, SALS))


# =========================================================================
# 4. Sea-water permittivity vs the Recommendation (eqs 5-27)
# =========================================================================
class TestSeaWaterPermittivity:
    def test_cpp_matches_recommendation_over_full_valid_range(self, cpp):
        got = cpp.eps(GRID)
        bad = []
        for (f, T, S), (re, im) in zip(GRID, got):
            rre, rim = ref_sea_water(f, T, S)
            e1, e2 = rel_err(re, float(rre)), rel_err(im, float(rim))
            if e1 > 1e-12 or e2 > 1e-12:
                bad.append((f, T, S, re, float(rre), im, float(rim)))
        assert not bad, f"{len(bad)}/{len(GRID)} points disagree with P.527-6 eq (15)/(16); first: {bad[:3]}"

    def test_python_mirror_matches_recommendation(self):
        """helpers.sea_permittivity is the oracle behind test_ns3_linkbudget.py; pin it."""
        bad = []
        for f, T, S in GRID:
            re, im = mirror_sea_permittivity(f, T, S)
            rre, rim = ref_sea_water(f, T, S)
            if rel_err(re, float(rre)) > 1e-12 or rel_err(im, float(rim)) > 1e-12:
                bad.append((f, T, S))
        assert not bad, f"mirror disagrees with P.527-6 at {len(bad)} points, e.g. {bad[:3]}"

    def test_salinity_zero_is_pure_water(self, cpp):
        """Rec. text after eq (27): pure water (5)-(7) is the S = 0 special case of (14)-(16)."""
        pts = [(f, T, 0.0) for f in FREQS for T in TEMPS]
        for (f, T, _), (re, im) in zip(pts, cpp.eps(pts)):
            pre, pim = ref_pure_water(f, T)
            assert re == pytest.approx(float(pre), rel=1e-12), (f, T)
            assert im == pytest.approx(float(pim), rel=1e-12, abs=1e-15), (f, T)

    def test_static_permittivity_is_eq8(self, cpp):
        """f -> 0, S = 0: eps' -> eps_s = 77.66 + 103.3*Theta.  At 20 C: Theta = 0.0233668770...,
        eps_s = 80.0737983967... (hand-checkable from eqs 8 and 11)."""
        (re, _), = cpp.eps([(1e-9, 20.0, 0.0)])
        assert re == pytest.approx(80.07379839672522, rel=1e-12)

    def test_high_frequency_limit_is_eps_inf_s(self, cpp):
        """f -> infinity: eps' -> eps_inf,s (eq 21)."""
        for T, S in [(20.0, 35.0), (0.0, 35.0), (10.0, 20.0)]:
            (re, _), = cpp.eps([(1e9, T, S)])
            eps_inf = float(_water_terms(T)[2])
            expect = eps_inf * (1 + S * (-2.04265e-3 + 1.57883e-4 * T))
            assert re == pytest.approx(expect, rel=1e-6), (T, S)

    def test_ionic_conductivity_anchor_15C_35ppt(self, cpp):
        """Extract sigma from the 18*sigma/f term at very low f and compare with eq (22):
        at T = 15 C, R_T15 = 1 exactly, sigma_35(15 C) = 4.2913986829 S/m (eq 23) and
        R_15(35 ppt) = 0.99998935788 (eq 24), so sigma = 4.29135302 S/m."""
        f = 1e-6
        (_, im), = cpp.eps([(f, 15.0, 35.0)])
        sigma = im * f / 18.0
        assert sigma == pytest.approx(4.2913986829374995 * 0.9999893578778709, rel=1e-8)

    def test_conductivity_term_consistent_with_eq_3a(self, cpp):
        """Eq (2)/(3a): conduction loss = sigma/(2*pi*f*eps0) = sigma/(0.05563 f_GHz).  The
        Rec.'s eq (16) writes this as 18*sigma/f; the constants agree to < 0.2 %.  Guards
        against someone changing the literal 18."""
        f = 1e-6
        (_, im), = cpp.eps([(f, 20.0, 35.0)])
        sigma = float(ref_sigma_sw(20.0, 35.0))
        assert im * f == pytest.approx(sigma / 0.05563, rel=2e-3)

    def test_inputs_are_clamped_to_the_valid_range(self, cpp):
        """Rec. Sec. 5.1: valid for -4..40 C, 0..40 ppt.  The code documents clamping rather
        than extrapolating; pin that behaviour."""
        f = 16.0
        pairs = [((f, 60.0, 35.0), (f, 40.0, 35.0)), ((f, -10.0, 35.0), (f, -4.0, 35.0)),
                 ((f, 20.0, 60.0), (f, 20.0, 40.0)), ((f, 20.0, -5.0), (f, 20.0, 0.0))]
        for a, b in pairs:
            ra, rb = cpp.eps([a, b])
            assert ra == pytest.approx(rb, rel=1e-15), (a, b)

    def test_losses_are_non_negative_and_finite(self, cpp):
        for (f, T, S), (re, im) in zip(GRID, cpp.eps(GRID)):
            assert math.isfinite(re) and math.isfinite(im)
            assert re > 1.0 and im > 0.0, (f, T, S, re, im)

    def test_regression_pins(self, cpp):
        """Pins derived from the Rec's equations with mpmath (NOT ITU-published): the repo's
        two bands under the default scenario (L 1.5 GHz, Ku 16 GHz, 20 C, 35 ppt), the
        Fig. 3 condition (0 C), pure water at Ku, and 60 GHz."""
        keys = list(P527_PINS)
        for k, (re, im) in zip(keys, cpp.eps(keys)):
            pre, pim = P527_PINS[k]
            assert re == pytest.approx(pre, rel=1e-9), k
            assert im == pytest.approx(pim, rel=1e-9), k


# (f_GHz, T_C, S_ppt) -> (eps', eps'');  eps_sw = eps' - j eps''
P527_PINS = {
    (1.5, 20.0, 35.0): (71.327250730273, 62.284334764837),
    (16.0, 20.0, 35.0): (46.894267573297, 37.687127282996),
    (1.5, 0.0, 35.0): (76.860183356419, 46.001602712877),
    (16.0, 0.0, 35.0): (24.996632458394, 35.812583770099),
    (16.0, 20.0, 0.0): (44.877764922158, 37.336255639489),
    (60.0, 20.0, 35.0): (12.104845905188, 22.376487863995),
}


# =========================================================================
# 5. Fresnel reflection coefficient vs Rec. Sec. 6 (eqs 91-96)
# =========================================================================
PSI = [1e-6, 1e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 0.7, 1.2, math.pi / 2]   # grazing angle, rad
REFL_GRID = list(itertools.product(PSI, [1.5, 16.0], [True, False], [0.0, 20.0, 40.0], [0.0, 35.0]))


class TestFresnelReflection:
    def test_cpp_matches_eqs_91_92(self, cpp):
        got = cpp.refl(REFL_GRID)
        bad = []
        for (psi, f, v, T, S), (gr, gi) in zip(REFL_GRID, got):
            ere, eim = ref_sea_water(f, T, S)
            r = ref_fresnel_grazing(psi, ere, eim, v)
            if abs(complex(gr, gi) - complex(r)) > 1e-12:
                bad.append((psi, f, v, T, S, complex(gr, gi), complex(r)))
        assert not bad, f"{len(bad)}/{len(REFL_GRID)} disagree with eq (91)/(92); first: {bad[:2]}"

    def test_passivity_and_valid_emissivity(self, cpp):
        """Eq (90): emissivity = 1 - |r|^2 must lie in [0, 1] for every passive medium.  A
        wrong sign convention on Im(eps) (e^{+jwt} vs e^{-jwt}) typically breaks this."""
        for case, (gr, gi) in zip(REFL_GRID, cpp.refl(REFL_GRID)):
            e = 1.0 - (gr * gr + gi * gi)
            assert -1e-12 <= e <= 1.0 + 1e-12, (case, e)

    def test_normal_incidence_rv_equals_minus_rh_and_rc_zero(self, cpp):
        """Eqs (94)-(96): at normal incidence r_v = -r_h, so r_c = (r_v + r_h)/2 = 0."""
        cases_v = [(math.pi / 2, f, True, 20.0, 35.0) for f in (1.5, 16.0)]
        cases_h = [(math.pi / 2, f, False, 20.0, 35.0) for f in (1.5, 16.0)]
        for (rv_re, rv_im), (rh_re, rh_im) in zip(cpp.refl(cases_v), cpp.refl(cases_h)):
            assert rv_re == pytest.approx(-rh_re, abs=1e-12)
            assert rv_im == pytest.approx(-rh_im, abs=1e-12)

    def test_normal_incidence_closed_form_eq_95(self, cpp):
        """Eq (95): r_h(normal) = (1 - sqrt(eps)) / (1 + sqrt(eps))."""
        for f in (1.5, 16.0):
            ere, eim = ref_sea_water(f, 20.0, 35.0)
            s = mpsqrt(mpc(ere, -eim))
            expect = (1 - s) / (1 + s)
            (gr, gi), = cpp.refl([(math.pi / 2, f, False, 20.0, 35.0)])
            assert abs(complex(gr, gi) - complex(expect)) < 1e-12

    def test_grazing_limit_both_polarisations_go_to_minus_one(self, cpp):
        """psi -> 0: both r_v and r_h -> -1 (the physical basis of the two-ray lobing)."""
        for v in (True, False):
            (gr, gi), = cpp.refl([(1e-9, 16.0, v, 20.0, 35.0)])
            assert complex(gr, gi) == pytest.approx(-1.0, abs=1e-6)


# =========================================================================
# 6. Mirror integrity (NOT a P.527 check): helpers.two_ray_db vs the compiled C++
#    test_ns3_linkbudget.py trusts helpers.two_ray_db as its oracle; this pins that oracle
#    to the real code so the oracle cannot silently drift.
# =========================================================================
TWO_RAY_GRID = [(d, h1, h2, f, ss, v, T, S)
                for d in (50.0, 500.0, 3000.0, 20000.0, 60000.0)
                for h1 in (5.0, 30.0, 150.0)
                for h2 in (5.0, 300.0)
                for f in (1.5, 16.0)
                for ss in (0, 3, 6)
                for v in (True, False)
                for T, S in ((20.0, 35.0), (5.0, 10.0))]


class TestMirrorIntegrity:
    def test_python_two_ray_matches_compiled_cpp(self, cpp):
        got = cpp.tworay(TWO_RAY_GRID)
        bad = []
        for (d, h1, h2, f, ss, v, T, S), (loss,) in zip(TWO_RAY_GRID, got):
            py = float(mirror_two_ray_db(d, h1, h2, f, ss, v, T, S))
            if abs(py - loss) > 1e-6:
                bad.append(((d, h1, h2, f, ss, v, T, S), loss, py))
        assert not bad, (f"helpers.two_ray_db has drifted from uav_mesh_eval.cc at {len(bad)}/"
                         f"{len(TWO_RAY_GRID)} points; first: {bad[:2]}")