#!/usr/bin/env python3
"""Build Tables 2-5 (and paired statistics) from the benchmark and ns-3 replication outputs.

  python testing/make_tables.py \
      --bench results/placement_benchmark.csv \
      --arm midpoint=results/midpoint --arm grid=results/grid \
      --arm de1=results/de1 --arm de10=results/de10 --arm de60=results/de60 \
      --out results/tables

--bench : CSV from testing/placement_benchmark.py            -> Tables 2 and 4 (proxy scores)
--arm   : name=<work-dir of one run_replications.py arm>     -> Tables 3 and 5 (ns-3 metrics)
          (each dir must contain replication_ship_metrics.csv)

Conventions (state these in the report):
* Every cell is a mean over scenarios.  For ns-3 metrics each scenario is first reduced to
  the mean over its ships, so every scenario has equal weight.
* Signal and ETX are -inf / inf for a ship with no direct UAV link at some step.  Those
  ships are excluded from the signal/ETX means and reported separately as "link availability"
  (% of ships with a direct UAV link at every step).
* Only scenarios present in EVERY arm are used, so all comparisons are paired.
* UAV position is reported relative to the fleet centroid (absolute E/N differ by scenario
  and their mean is meaningless).
* "vs reference": paired difference (arm - reference) with 95 % t-interval and a Wilcoxon
  signed-rank p-value.  Lower is better for score, link length, latency, ETX.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def paired(a, b):
    """Return (mean diff, ci_lo, ci_hi, p_wilcoxon, n) for a - b (aligned arrays, NaN-pairs dropped)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    n = len(d)
    if n < 2:
        return (float(d.mean()) if n else math.nan, math.nan, math.nan, math.nan, n)
    m, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if sd > 0 else 0.0
    if np.all(d == 0):
        p = 1.0
    else:
        try:
            p = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            p = math.nan
    return float(m), float(m - half), float(m + half), p, n


def fmt_p(p):
    return "n/a" if not math.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def md_table(title, header, rows):
    out = [f"\n### {title}\n", "| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------ proxy tables (2, 4)
def proxy_tables(bench_csv, out, text):
    rows = read(bench_csv)
    by = defaultdict(dict)                        # method -> scenario -> row
    for r in rows:
        by[r["method"]][int(r["scenario"])] = r
    methods = list(by)
    common = sorted(set.intersection(*[set(v) for v in by.values()]))
    n = len(common)
    col = lambda m, k: np.array([float(by[m][s][k]) for s in common])
    de = [m for m in methods if m.startswith("de")]
    de.sort(key=lambda m: float(m[2:]))
    label = lambda m: "Grid" if m == "grid" else "Midpoint" if m == "midpoint" else f"DE ({m[2:]}s)"

    def table(title, ref, name):
        order = [m for m in [ref] + de if m in by]
        body, csvrows = [], []
        for m in order:
            sc = col(m, "score")
            pos = (f"dE={col(m, 'd_east_from_centroid_m').mean() / 1000:.2f} km, "
                   f"dN={col(m, 'd_north_from_centroid_m').mean() / 1000:.2f} km, "
                   f"U={col(m, 'alt_m').mean():.0f} m")
            body.append([label(m), f"{sc.mean():.2f} (median {np.median(sc):.2f})",
                         f"{col(m, 'uav_link_km').mean():.2f} km", pos,
                         f"{col(m, 'compute_s').mean():.2f} s", f"{col(m, 'evaluations').mean():.0f}"])
            csvrows.append([label(m), sc.mean(), np.median(sc), col(m, "uav_link_km").mean(),
                            col(m, "d_east_from_centroid_m").mean(), col(m, "d_north_from_centroid_m").mean(),
                            col(m, "alt_m").mean(), col(m, "compute_s").mean(), col(m, "evaluations").mean()])
        hdr = ["", "Score (mean)", "UAV link length (mean)", "Position rel. to fleet centroid (mean)",
               "Compute time (mean)", "Evaluations (mean)"]
        text.append(md_table(f"{title} (n = {n} scenarios)", hdr, body))
        save_csv(out / f"{name}.csv", ["method", "score_mean", "score_median", "link_km_mean",
                                        "dE_from_centroid_m", "dN_from_centroid_m", "alt_m_mean",
                                        "compute_s_mean", "evaluations_mean"], csvrows)
        prow = []
        for m in de:
            sm, lo, hi, p, k = paired(col(m, "score"), col(ref, "score"))
            ties = int(np.sum(np.isclose(col(m, "score"), col(ref, "score"), rtol=0, atol=1e-9)))
            better = int(np.sum(col(m, "score") < col(ref, "score") - 1e-9))
            worse = int(np.sum(col(m, "score") > col(ref, "score") + 1e-9))
            prow.append([label(m), f"{sm:+.2f} [{lo:+.2f}, {hi:+.2f}]", fmt_p(p),
                         f"{better} / {ties} / {worse}"])
        text.append(md_table(f"{name}: DE minus {label(ref)} score, paired",
                             ["DE budget", "mean diff [95% CI]", "Wilcoxon p", "better / equal / worse (scenarios)"],
                             prow))

    if "grid" in by:
        table("Table 2: DE optimiser vs brute-force grid, proxy scores", "grid", "table2_grid_vs_de_proxy")
    if "midpoint" in by:
        table("Table 4: DE optimiser vs midpoint placement, proxy scores", "midpoint",
              "table4_midpoint_vs_de_proxy")


# ------------------------------------------------------------------ ns-3 tables (3, 5)
METRICS = [("min_model_path_rx_dbm", "Signal strength (dBm)", "%.1f", True),
           ("latency_ms", "Latency (ms)", "%.3f", False),
           ("throughput_kbps", "Throughput (kbps)", "%.2f", False),
           ("max_model_path_etx", "ETX", "%.3f", True),
           ("pdr_pct", "PDR (%)", "%.2f", False)]


def ns3_tables(arms, out, text):
    per = {}                                     # arm -> scenario -> metric -> value
    for name, d in arms.items():
        rows = read(Path(d) / "replication_ship_metrics.csv")
        sc = defaultdict(lambda: defaultdict(list))
        for r in rows:
            s = int(r["replication"])
            for key, _, _, finite_only in METRICS:
                v = float(r[key])
                if finite_only and not math.isfinite(v):
                    continue
                sc[s][key].append(v)
            sc[s]["_ships"].append(1)
            sc[s]["_linked"].append(1 if math.isfinite(float(r["min_model_path_rx_dbm"])) else 0)
        per[name] = {s: {k: (float(np.mean(v)) if v else math.nan) for k, v in m.items()}
                     for s, m in sc.items()}
    common = sorted(set.intersection(*[set(v) for v in per.values()]))
    n = len(common)
    col = lambda a, k: np.array([per[a][s].get(k, math.nan) for s in common])
    label = lambda m: "Grid" if m == "grid" else "Midpoint" if m == "midpoint" else f"DE ({m[2:]}s)"
    de = sorted([a for a in arms if a.startswith("de")], key=lambda m: float(m[2:]))

    def table(title, ref, name):
        order = [a for a in [ref] + de if a in arms]
        body, csvrows = [], []
        for a in order:
            cells = [label(a)]
            csvr = [label(a)]
            for key, _, f, _ in METRICS:
                v = col(a, key)
                cells.append(f % np.nanmean(v))
                csvr.append(np.nanmean(v))
            avail = 100.0 * col(a, "_linked").mean()
            cells.append(f"{avail:.1f} %")
            csvr.append(avail)
            body.append(cells)
            csvrows.append(csvr)
        hdr = [""] + [m[1] for m in METRICS] + ["Ships linked at all steps"]
        text.append(md_table(f"{title} (n = {n} paired scenarios)", hdr, body))
        save_csv(out / f"{name}.csv", ["method"] + [m[0] for m in METRICS] + ["link_availability_pct"],
                 csvrows)
        prow = []
        for a in de:
            row = [label(a)]
            for key, nm, f, _ in METRICS:
                dm, lo, hi, p, k = paired(col(a, key), col(ref, key))
                row.append(f"{dm:+.3g} [{lo:+.3g}, {hi:+.3g}] p={fmt_p(p)}")
            dm, lo, hi, p, k = paired(col(a, "_linked"), col(ref, "_linked"))
            row.append(f"{100 * dm:+.1f} pp p={fmt_p(p)}")
            prow.append(row)
        text.append(md_table(f"{name}: DE minus {label(ref)}, paired (mean diff [95% CI], Wilcoxon p)",
                             [""] + [m[1] for m in METRICS] + ["Link availability"], prow))

    if "grid" in arms:
        table("Table 3: DE optimiser vs brute-force grid, ns-3 metrics", "grid", "table3_grid_vs_de_ns3")
    if "midpoint" in arms:
        table("Table 5: DE optimiser vs midpoint placement, ns-3 metrics", "midpoint",
              "table5_midpoint_vs_de_ns3")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", help="placement_benchmark.csv (Tables 2, 4)")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR",
                    help="ns-3 arm: midpoint | grid | de<seconds> = run_replications work-dir")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = []
    if args.bench:
        proxy_tables(args.bench, out, text)
    arms = dict(a.split("=", 1) for a in args.arm)
    if arms:
        ns3_tables(arms, out, text)
    report = "\n".join(text)
    (out / "tables.md").write_text(report + "\n")
    print(report)
    print(f"\n[make_tables] wrote {out}/tables.md and per-table CSVs")


if __name__ == "__main__":
    main()