#!/usr/bin/env python3
"""
evaluate_objectives.py -- score ns-3 runs against the supervisor's objectives.

Reads the raw CSVs that uav_mesh_eval.cc writes (<prefix>_linkbudget.csv, _datatransfer.csv,
_telemetry.csv, ...) plus each run's export nodes.csv, and prints PASS / FAIL / N/M for every
objective, with confidence intervals across replications and, when both placements are present,
a PAIRED optimized-vs-midpoint comparison.

OBJECTIVES (defaults; override with flags)                         comparison
  Received power   >= -100 dBm  on every link in use                  >=
  Latency          <  50 ms     one-way, mesh only (SATCOM ignored)   <
  Ku throughput    >  1 Gbps over 30 nodes = 33.33 Mbps per ship      >   (scaled to the ships you run)
  L-band throughput > 5 Mbps per ship                                 >
  ETX              <  2.2 per route                                   <

THREE OUTCOMES, NOT TWO
  PASS / FAIL  the objective was measured and met / missed.
  N/M          NOT MEASURABLE: the run cannot answer the question. Examples: the offered traffic was
               below the throughput target (so a "pass" would only reflect the traffic generator), or
               the run produced no L-band saturating flow. N/M is never counted as a pass.

WHERE EACH NUMBER COMES FROM, AND HOW FAR TO TRUST IT
  Received power  <prefix>_linkbudget.csv rx_dbm (mean received power, before fast fading) for
                  every link of the ship (--signal-scope all) or only its links to the UAV (uav). A ship
                  with no link at some time step FAILS (nothing to receive).
  Latency         mean one-way delay of the ship->UAV data flow and of the ship's telemetry. The
                  simulator fixes propagation at 1 us, so real propagation (<~0.2 ms at 50 km) is not
                  included. Only MEANS are written by ns-3, so this is weaker than a percentile.
  Ku throughput   <prefix>_datatransfer.csv throughput_kbps (ship->UAV flow = Ku).
  L throughput    <prefix>_lthroughput.csv (same columns as datatransfer). Absent until the L-band
                  saturating flow exists -> N/M.
  ETX             <prefix>_etx.csv (MEASURED, one row per route) when present; otherwise a MODELED
                  estimate 1/(1-outage_prob) of the ship->UAV Ku link. Output says which.

USAGE
  # replications written by run_replications.py (single or --compare-midpoint layout)
  python3 evaluate_objectives.py --ns3-root ~/ns-3-dev --work-dir replications --out-prefix rep

  # one run from run_pipeline.py
  python3 evaluate_objectives.py --ns3-root ~/ns-3-dev --run ns3_export:with_uav:optimized

  # save tables for the report
  python3 evaluate_objectives.py --ns3-root ~/ns-3-dev --work-dir replications \\
      --csv-out per_ship.csv --md-out summary.md
"""
import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

PASS, FAIL, NM = "PASS", "FAIL", "N/M"
OBJECTIVES = ("signal", "latency", "ku_throughput", "l_throughput", "etx")
LABELS = {"signal": "Received power", "latency": "Latency", "ku_throughput": "Ku throughput",
          "l_throughput": "L-band throughput", "etx": "ETX (per route)"}


@dataclass
class Objectives:
    signal_min_dbm: float = -100.0          # >=
    latency_max_ms: float = 50.0            # <
    ku_total_mbps_ref: float = 1000.0       # >  ... over ku_ref_nodes nodes
    ku_ref_nodes: int = 30
    l_per_ship_mbps: float = 5.0            # >
    etx_max: float = 2.2                    # <
    signal_scope: str = "all"               # all | uav
    etx_stat: str = "max"                   # max | mean (over a ship's routes / time steps)
    ku_offered_mbps: float = None           # known offered Ku rate per ship (else inferred)
    l_offered_mbps: float = None

    @property
    def ku_per_ship_mbps(self):
        return self.ku_total_mbps_ref / self.ku_ref_nodes

    def ku_total_target(self, n_ships):
        return self.ku_per_ship_mbps * n_ships


# ------------------------------------------------------------------ small helpers
def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def maybe_csv(path):
    return read_csv(path) if Path(path).is_file() else None


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion; (nan, nan) if n == 0."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mean_ci(values, conf=0.95):
    """(mean, half_width) with a Student-t interval; half_width is nan for n < 2."""
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return (math.nan, math.nan)
    m = statistics.mean(vals)
    if len(vals) < 2:
        return (m, math.nan)
    try:
        from scipy import stats
        t = stats.t.ppf(0.5 + conf / 2, len(vals) - 1)
    except ImportError:                      # normal approximation as a fallback
        t = 1.96
    return (m, t * statistics.stdev(vals) / math.sqrt(len(vals)))


def fmt(x, nd=2, unit=""):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if x == math.inf:
        return "inf"
    if x == -math.inf:
        return "no link"
    return f"{x:.{nd}f}{unit}"


# ------------------------------------------------------------------ run discovery
@dataclass
class Run:
    replication: int
    placement: str
    prefix: str
    ns3_root: Path
    export_dir: Path


def discover_runs(work_dir, out_prefix, ns3_root, label="run"):
    """Find runs written by run_replications.py in either layout:
         exports/rep<i>/nodes.csv                  -> prefix <out_prefix><i>
         exports/rep<i>/<placement>/nodes.csv      -> prefix <out_prefix><i>_<placement>"""
    runs = []
    exports = Path(work_dir) / "exports"
    for d in sorted(exports.glob("rep*"), key=lambda p: int(p.name[3:]) if p.name[3:].isdigit() else 0):
        if not d.name[3:].isdigit():
            continue
        i = int(d.name[3:])
        if (d / "nodes.csv").is_file():
            runs.append(Run(i, label, f"{out_prefix}{i}", ns3_root, d))
        else:
            for sub in sorted(p for p in d.iterdir() if (p / "nodes.csv").is_file()):
                runs.append(Run(i, sub.name, f"{out_prefix}{i}_{sub.name}", ns3_root, sub))
    return runs


def parse_run_spec(spec, ns3_root, index):
    """EXPORT_DIR:PREFIX[:PLACEMENT[:REPLICATION]]"""
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"--run expects EXPORT_DIR:PREFIX[:PLACEMENT[:REP]], got {spec!r}")
    placement = parts[2] if len(parts) > 2 and parts[2] else "run"
    rep = int(parts[3]) if len(parts) > 3 else index
    return Run(rep, placement, parts[1], ns3_root, Path(parts[0]))


# ------------------------------------------------------------------ per-run scoring
@dataclass
class ShipScore:
    ship: int
    signal_dbm: float = math.nan
    isolated_steps: int = 0
    data_latency_ms: float = math.nan
    tel_latency_ms: float = math.nan
    ku_mbps: float = math.nan
    ku_offered_mbps: float = math.nan
    l_mbps: float = math.nan
    l_offered_mbps: float = math.nan
    etx: float = math.nan
    etx_source: str = "none"
    verdict: dict = field(default_factory=dict)


@dataclass
class RunScore:
    run: Run
    n_ships: int
    ships: list
    verdict: dict
    ku_total_mbps: float
    ku_total_target: float
    notes: list

    def ship_pass_rate(self, obj):
        meas = [s.verdict[obj] for s in self.ships if s.verdict[obj] != NM]
        return (sum(v == PASS for v in meas) / len(meas)) if meas else math.nan


def _rx_per_ship(linkbudget, ships, uav, steps, scope):
    per = {s: {st: [] for st in steps} for s in ships}
    for r in linkbudget:
        i, j, st, rx = int(r["i"]), int(r["j"]), int(r["step"]), float(r["rx_dbm"])
        for a, b in ((i, j), (j, i)):
            if a in per and st in per[a] and (scope == "all" or b == uav):
                per[a][st].append(rx)
    out = {}
    for s in ships:
        allrx = [v for lst in per[s].values() for v in lst]
        isolated = sum(1 for lst in per[s].values() if not lst)
        out[s] = (min(allrx) if allrx else -math.inf, isolated)
    return out


def _modeled_etx(linkbudget, ships, uav, steps, stat):
    per = {s: {} for s in ships}
    for r in linkbudget:
        if r["band"] != "Ku":
            continue
        i, j = int(r["i"]), int(r["j"])
        if uav not in (i, j):
            continue
        s = j if i == uav else i
        if s not in per:
            continue
        out = float(r["outage_prob"])
        per[s][int(r["step"])] = math.inf if out >= 1.0 else 1.0 / (1.0 - out)
    res = {}
    for s in ships:
        vals = [per[s].get(st, math.inf) for st in steps]      # a step with no Ku link = no route
        res[s] = max(vals) if stat == "max" else statistics.mean(vals)
    return res


def _measured_etx(etx_rows, ships, stat):
    """One value per ship from <prefix>_etx.csv rows (src,dst,...,route_etx). Rows whose ETX is
    NaN were not measurable (no traffic on a link of the route) and are ignored; a ship with
    ONLY such rows has no measurement (NaN -> N/M). A route with no path at all is +inf."""
    by = {}
    for r in etx_rows:
        by.setdefault(int(r["src"]), []).append(float(r["route_etx"]))
    out = {}
    for s in ships:
        vals = [v for v in by.get(s, []) if not math.isnan(v)]
        if not vals:
            out[s] = math.nan if s in by else math.inf    # no rows at all = no route
        else:
            out[s] = max(vals) if stat == "max" else statistics.mean(vals)
    return out


def score_run(run, obj):
    p = run.ns3_root / run.prefix
    nodes = read_csv(run.export_dir / "nodes.csv")
    ids = sorted({int(r["node"]) for r in nodes})
    uav = ids[-1]
    ships = ids[:-1]
    steps = sorted({int(r["step"]) for r in nodes})
    notes = []

    lb = read_csv(f"{p}_linkbudget.csv")
    data = {int(r["ship"]): r for r in read_csv(f"{p}_datatransfer.csv")}
    tel = {int(r["ship"]): r for r in (maybe_csv(f"{p}_telemetry.csv") or [])}
    lth = {int(r["ship"]): r for r in (maybe_csv(f"{p}_lthroughput.csv") or [])}
    etx_rows = maybe_csv(f"{p}_etx.csv")

    rx = _rx_per_ship(lb, ships, uav, steps, obj.signal_scope)
    if etx_rows is not None:
        etx_val = _measured_etx(etx_rows, ships, obj.etx_stat)
        etx_src = "measured"
    else:
        etx_val = _modeled_etx(lb, ships, uav, steps, obj.etx_stat)
        etx_src = "modeled"
        notes.append("ETX is MODELED from the link budget (no <prefix>_etx.csv); not a measurement")

    ku_target, l_target = obj.ku_per_ship_mbps, obj.l_per_ship_mbps
    scores = []
    for s in ships:
        sc = ShipScore(ship=s, signal_dbm=rx[s][0], isolated_steps=rx[s][1], etx=etx_val[s],
                       etx_source=etx_src)
        d = data.get(s)
        if d is not None:
            rxp, off = int(d["rx_pkts"]), int(d["offered_pkts"])
            sc.data_latency_ms = float(d["mean_delay_ms"]) if rxp else math.inf
            sc.ku_mbps = float(d["throughput_kbps"]) / 1000.0
            if obj.ku_offered_mbps is not None:
                sc.ku_offered_mbps = obj.ku_offered_mbps
            elif rxp and off:
                sc.ku_offered_mbps = sc.ku_mbps * off / rxp
        t = tel.get(s)
        if t is not None:
            sc.tel_latency_ms = float(t["mean_delay_ms"]) if int(t["rx_receptions"]) else math.inf
        l = lth.get(s)
        if l is not None:
            rxp, off = int(l["rx_pkts"]), int(l["offered_pkts"])
            sc.l_mbps = float(l["throughput_kbps"]) / 1000.0
            if obj.l_offered_mbps is not None:
                sc.l_offered_mbps = obj.l_offered_mbps
            elif rxp and off:
                sc.l_offered_mbps = sc.l_mbps * off / rxp

        v = sc.verdict
        v["signal"] = PASS if (sc.isolated_steps == 0 and sc.signal_dbm >= obj.signal_min_dbm) else FAIL
        lats = [x for x in (sc.data_latency_ms, sc.tel_latency_ms) if not math.isnan(x)]
        v["latency"] = NM if not lats else (PASS if all(x < obj.latency_max_ms for x in lats) else FAIL)
        v["ku_throughput"] = _throughput_verdict(sc.ku_mbps, sc.ku_offered_mbps, ku_target)
        v["l_throughput"] = _throughput_verdict(sc.l_mbps, sc.l_offered_mbps, l_target)
        v["etx"] = NM if math.isnan(sc.etx) else (PASS if sc.etx < obj.etx_max else FAIL)
        scores.append(sc)

    ku_total = sum(s.ku_mbps for s in scores if not math.isnan(s.ku_mbps))
    ku_total_target = obj.ku_total_target(len(ships))
    run_v = {}
    for name in OBJECTIVES:
        vs = [s.verdict[name] for s in scores]
        run_v[name] = NM if NM in vs else (PASS if all(x == PASS for x in vs) else FAIL)
    if run_v["ku_throughput"] != NM:        # objective is the AGGREGATE; per-ship is diagnostic
        run_v["ku_throughput"] = PASS if ku_total > ku_total_target else FAIL
    if run_v["ku_throughput"] == NM:
        notes.append(f"Ku throughput N/M: offered load below the {ku_target:.1f} Mbps/ship target "
                     f"(or no flow data)")
    if run_v["l_throughput"] == NM:
        notes.append("L-band throughput N/M: no saturating L-band flow in this run")
    return RunScore(run, len(ships), scores, run_v, ku_total, ku_total_target, notes)


def _throughput_verdict(measured, offered, target):
    """Strictly '>' the target. N/M unless the offered load actually reached the target."""
    if math.isnan(measured) or math.isnan(offered) or offered < target:
        return NM
    return PASS if measured > target else FAIL


# ------------------------------------------------------------------ aggregation / reporting
def _scalar(rs, key):
    """Run-level scalar used for confidence intervals and paired deltas."""
    ships = rs.ships
    if key == "worst_rx_dbm":
        return min(s.signal_dbm for s in ships)
    if key == "worst_latency_ms":
        v = [max(x for x in (s.data_latency_ms, s.tel_latency_ms) if not math.isnan(x))
             for s in ships if not (math.isnan(s.data_latency_ms) and math.isnan(s.tel_latency_ms))]
        return max(v) if v else math.nan
    if key == "ku_total_mbps":
        return rs.ku_total_mbps
    if key == "worst_etx":
        vals = [s.etx for s in ships if not math.isnan(s.etx)]
        return max(vals) if vals else math.nan
    raise KeyError(key)


SCALARS = {"signal": "worst_rx_dbm", "latency": "worst_latency_ms", "ku_throughput": "ku_total_mbps",
           "etx": "worst_etx"}


def summarise(scores, obj):
    """One block per placement: run-level verdict counts + CIs."""
    out = {}
    for placement in sorted({s.run.placement for s in scores}):
        rs = [s for s in scores if s.run.placement == placement]
        block = {"n_runs": len(rs)}
        for name in OBJECTIVES:
            verd = [r.verdict[name] for r in rs]
            k, nm = verd.count(PASS), verd.count(NM)
            n_meas = len(verd) - nm
            lo, hi = wilson(k, n_meas)
            rate_m, rate_hw = mean_ci([r.ship_pass_rate(name) for r in rs])
            entry = {"pass": k, "fail": verd.count(FAIL), "nm": nm, "wilson": (lo, hi),
                     "ship_rate": (rate_m, rate_hw)}
            if name in SCALARS:
                entry["scalar"] = mean_ci([_scalar(r, SCALARS[name]) for r in rs])
            block[name] = entry
        out[placement] = block
    return out


def paired(scores, a="optimized", b="midpoint"):
    """Per-replication differences a - b (positive = a is better for pass rates / higher-is-better)."""
    by = {}
    for s in scores:
        by.setdefault(s.run.replication, {})[s.run.placement] = s
    pairs = [(d[a], d[b]) for d in by.values() if a in d and b in d]
    out = {}
    for name in OBJECTIVES:
        deltas = [(x.ship_pass_rate(name) - y.ship_pass_rate(name)) for x, y in pairs]
        deltas = [d for d in deltas if math.isfinite(d)]
        entry = {"n": len(deltas), "rate_delta": mean_ci(deltas),
                 "wins": sum(d > 1e-12 for d in deltas), "losses": sum(d < -1e-12 for d in deltas),
                 "ties": sum(abs(d) <= 1e-12 for d in deltas)}
        if name in SCALARS:
            sd = [_scalar(x, SCALARS[name]) - _scalar(y, SCALARS[name]) for x, y in pairs]
            entry["scalar_delta"] = mean_ci([d for d in sd if math.isfinite(d)])
        out[name] = entry
    return out, len(pairs)


def render(scores, obj, a="optimized", b="midpoint"):
    L = []
    summ = summarise(scores, obj)
    n_ships = scores[0].n_ships
    L.append("OBJECTIVES  (targets scaled to %d ship%s)" % (n_ships, "" if n_ships == 1 else "s"))
    L.append(f"  Received power >= {obj.signal_min_dbm:g} dBm [{obj.signal_scope} links]   "
             f"Latency < {obj.latency_max_ms:g} ms   ETX < {obj.etx_max:g} [{obj.etx_stat}]")
    L.append(f"  Ku > {obj.ku_per_ship_mbps:.2f} Mbps/ship = {obj.ku_total_target(n_ships):.1f} Mbps total "
             f"({obj.ku_total_mbps_ref:g} Mbps / {obj.ku_ref_nodes} nodes)   "
             f"L-band > {obj.l_per_ship_mbps:g} Mbps/ship")
    for placement, block in summ.items():
        L.append("")
        L.append(f"== {placement}: {block['n_runs']} run(s) ==")
        L.append(f"{'objective':20s} {'PASS':>5s} {'FAIL':>5s} {'N/M':>4s}  {'run pass rate (95% Wilson)':28s} "
                 f"{'ship pass rate (mean +/- CI)':30s} key number (mean +/- CI)")
        for name in OBJECTIVES:
            e = block[name]
            meas = e["pass"] + e["fail"]
            wl = "n/a (not measurable)" if meas == 0 else \
                f"{100 * e['pass'] / meas:5.1f}%  [{100 * e['wilson'][0]:.0f}-{100 * e['wilson'][1]:.0f}]"
            sr = e["ship_rate"]
            srs = "n/a" if math.isnan(sr[0]) else f"{100 * sr[0]:5.1f}% +/- {fmt(100 * sr[1], 1)}"
            key = ""
            if "scalar" in e:
                unit = {"signal": " dBm", "latency": " ms", "ku_throughput": " Mbps", "etx": ""}[name]
                lab = {"signal": "worst link", "latency": "worst ship", "ku_throughput": "total",
                       "etx": "worst route"}[name]
                nd = 3 if name == "etx" else 2      # 2.202 must not print as 2.20 next to a '< 2.2' target
                key = f"{lab}: {fmt(e['scalar'][0], nd, unit)} +/- {fmt(e['scalar'][1], nd)}"
            L.append(f"{LABELS[name]:20s} {e['pass']:5d} {e['fail']:5d} {e['nm']:4d}  {wl:28s} {srs:30s} {key}")
    pr, npairs = paired(scores, a, b)
    if npairs:
        L.append("")
        L.append(f"== PAIRED  {a} - {b}   ({npairs} scenario pairs; same ship geometry in both) ==")
        L.append(f"{'objective':20s} {'ship pass-rate delta':26s} {'wins/losses/ties':18s} key-number delta")
        for name in OBJECTIVES:
            e = pr[name]
            if e["n"] == 0:
                L.append(f"{LABELS[name]:20s} {'n/a (not measurable)':26s}")
                continue
            m, hw = e["rate_delta"]
            kd = ""
            if "scalar_delta" in e:
                kd = f"{fmt(e['scalar_delta'][0], 2)} +/- {fmt(e['scalar_delta'][1], 2)}"
            L.append(f"{LABELS[name]:20s} {100 * m:+6.1f} pts +/- {fmt(100 * hw, 1):>5s}   "
                     f"{e['wins']:3d}/{e['losses']:3d}/{e['ties']:3d}        {kd}")
        L.append("  (a CI that excludes 0 means the difference is unlikely to be chance; with few "
                 "pairs the CI is wide)")
    notes = sorted({n for s in scores for n in s.notes})
    if notes:
        L.append("")
        L.append("NOTES")
        L.extend(f"  - {n}" for n in notes)
    return "\n".join(L)


def render_markdown(scores, obj):
    summ = summarise(scores, obj)
    L = []
    for placement, block in summ.items():
        L.append(f"**{placement}** ({block['n_runs']} runs)\n")
        L.append("| Objective | Pass | Fail | N/M | Run pass rate (95% CI) | Ship pass rate |")
        L.append("|---|---:|---:|---:|---|---|")
        for name in OBJECTIVES:
            e = block[name]
            meas = e["pass"] + e["fail"]
            wl = "n/a" if meas == 0 else f"{100 * e['pass'] / meas:.0f}% ({100 * e['wilson'][0]:.0f}-{100 * e['wilson'][1]:.0f}%)"
            sr = e["ship_rate"]
            L.append(f"| {LABELS[name]} | {e['pass']} | {e['fail']} | {e['nm']} | {wl} | "
                     f"{'n/a' if math.isnan(sr[0]) else f'{100 * sr[0]:.0f}%'} |")
        L.append("")
    return "\n".join(L)


def write_ship_csv(path, scores):
    cols = ["replication", "placement", "ship", "signal_dbm", "isolated_steps", "data_latency_ms",
            "tel_latency_ms", "ku_mbps", "ku_offered_mbps", "l_mbps", "l_offered_mbps", "etx",
            "etx_source"] + [f"{o}_verdict" for o in OBJECTIVES]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rs in scores:
            for s in rs.ships:
                w.writerow([rs.run.replication, rs.run.placement, s.ship, s.signal_dbm, s.isolated_steps,
                            s.data_latency_ms, s.tel_latency_ms, s.ku_mbps, s.ku_offered_mbps, s.l_mbps,
                            s.l_offered_mbps, s.etx, s.etx_source] + [s.verdict[o] for o in OBJECTIVES])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns3-root", required=True, help="folder where ns-3 wrote the <prefix>_*.csv files")
    ap.add_argument("--work-dir", help="run_replications.py --work-dir (contains exports/rep*)")
    ap.add_argument("--out-prefix", default="rep", help="run_replications.py --out-prefix (default rep)")
    ap.add_argument("--run", action="append", default=[], metavar="EXPORT_DIR:PREFIX[:PLACEMENT[:REP]]",
                    help="score one run directly (repeatable)")
    ap.add_argument("--label", default="run", help="placement label when the layout has none")
    ap.add_argument("--compare", nargs=2, default=("optimized", "midpoint"), metavar=("A", "B"))
    g = ap.add_argument_group("objectives")
    g.add_argument("--signal-min-dbm", type=float, default=-100.0)
    g.add_argument("--latency-max-ms", type=float, default=50.0)
    g.add_argument("--ku-total-mbps", type=float, default=1000.0, help="Ku target at --ku-ref-nodes nodes")
    g.add_argument("--ku-ref-nodes", type=int, default=30)
    g.add_argument("--l-mbps", type=float, default=5.0)
    g.add_argument("--etx-max", type=float, default=2.2)
    g.add_argument("--signal-scope", choices=("all", "uav"), default="all")
    g.add_argument("--etx-stat", choices=("max", "mean"), default="max")
    g.add_argument("--ku-offered-mbps", type=float, default=None,
                   help="known offered Ku rate per ship (otherwise inferred from throughput/PDR)")
    g.add_argument("--l-offered-mbps", type=float, default=None)
    ap.add_argument("--csv-out", help="write per-ship rows here")
    ap.add_argument("--md-out", help="write a Markdown summary here")
    args = ap.parse_args(argv)

    obj = Objectives(args.signal_min_dbm, args.latency_max_ms, args.ku_total_mbps, args.ku_ref_nodes,
                     args.l_mbps, args.etx_max, args.signal_scope, args.etx_stat,
                     args.ku_offered_mbps, args.l_offered_mbps)
    ns3_root = Path(args.ns3_root).expanduser().resolve()
    runs = []
    if args.work_dir:
        runs += discover_runs(Path(args.work_dir).expanduser(), args.out_prefix, ns3_root, args.label)
    runs += [parse_run_spec(s, ns3_root, i) for i, s in enumerate(args.run)]
    if not runs:
        ap.error("no runs found: give --work-dir (with exports/rep*) or --run EXPORT_DIR:PREFIX")

    scores, skipped = [], []
    for r in runs:
        try:
            scores.append(score_run(r, obj))
        except FileNotFoundError as e:
            skipped.append(f"{r.prefix}: missing {Path(e.filename).name}")
    if not scores:
        sys.exit("no run could be scored. Missing files:\n  " + "\n  ".join(skipped))
    print(render(scores, obj, *args.compare))
    if skipped:
        print("\nSKIPPED (missing ns-3 output): " + "; ".join(skipped))
    if args.csv_out:
        write_ship_csv(args.csv_out, scores)
        print(f"\nper-ship rows -> {args.csv_out}")
    if args.md_out:
        Path(args.md_out).write_text(render_markdown(scores, obj))
        print(f"markdown summary -> {args.md_out}")


if __name__ == "__main__":
    main()
