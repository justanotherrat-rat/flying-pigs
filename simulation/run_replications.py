#!/usr/bin/env python3
"""
run_replications.py -- run the random-placement + ns-3 pipeline N times and
report mean/std across replications.

WHAT THIS DOES
--------------
Monte Carlo over the whole pipeline: for i in 0..n-1,
  1. seed the RNGs for this replication (Python's and ns-3's, independently),
  2. generate a FRESH random ship placement + UAV optimisation (the notebook
     stage), written to its own export directory so replications never clobber
     each other's input files,
  3. run the ns-3 scenario against that export (reusing run_pipeline.py's
     sync_scratch_file/run_ns3 helpers),
  4. parse that replication's *_telemetry.csv / *_datatransfer.csv / *_mac.csv
     into one summary row.
After all N runs, it prints and writes a table of each metric's mean, sample
standard deviation, min and max ACROSS replications - this is what tells you
whether a result (e.g. "the UAV lifts PDR from 60% to 93%") holds up over
different random ship geometries, or whether it was a property of one
particular placement.

Each replication is an INDEPENDENT ship placement + full ns-3 run, so this is
the slow, correct way to get a confidence interval - not an approximation.
With a 60-minute scenario and a realistic ns-3 scratch build, each replication
can take anywhere from under a minute to several minutes depending on your
--sim-time/--n-ships and machine; n=20-30 is a reasonable starting point for
seeing whether an effect is stable, not a statistically rigorous sample size.

USAGE
-----
  python3 run_replications.py --ns3-root ~/ns-3-dev --n 20 --out-prefix with_uav \\
      --useUav=1 --seed 1000

  # anything not recognised here (--seaState, --fadeKssDb, --routing, ...) is
  # forwarded to uav_mesh_eval.cc for EVERY replication, exactly like
  # run_pipeline.py
  python3 run_replications.py --ns3-root ~/ns-3-dev --n 20 --out-prefix stress \\
      --useUav=1 --seaState=6 --fadeKssDb=-100

Results land in <work-dir>/replication_results.csv (one row per replication)
and <work-dir>/replication_summary.csv (one row per metric, aggregated).
"""
import argparse
import csv
import importlib
import statistics
import sys
from pathlib import Path

# run_pipeline.py must be importable (same directory, by default)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline  # noqa: E402 - needs the sys.path insert above first


def read_csv_dicts(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarise_replication(ns3_root: Path, out_prefix: str) -> dict:
    """Reads one replication's *_telemetry.csv / *_datatransfer.csv /
    *_mac.csv (written by uav_mesh_eval.cc into ns3_root) and reduces each to
    the handful of scalars worth tracking across replications. Per-ship rows
    are combined by summing counts (so the result is a true fleet-wide PDR,
    not an average of five PDR percentages) and weighting delay/hop averages
    by each ship's received-packet count."""

    def weighted(rows, rx_key, val_key):
        totalW = sum(int(r[rx_key]) for r in rows)
        if totalW == 0:
            return 0.0
        return sum(int(r[rx_key]) * float(r[val_key]) for r in rows) / totalW

    tel = read_csv_dicts(ns3_root / f"{out_prefix}_telemetry.csv")
    rx = sum(int(r["rx_receptions"]) for r in tel)
    possible = sum(int(r["possible_receptions"]) for r in tel)

    dat = read_csv_dicts(ns3_root / f"{out_prefix}_datatransfer.csv")
    drx = sum(int(r["rx_pkts"]) for r in dat)
    doff = sum(int(r["offered_pkts"]) for r in dat)

    mac = {r["band"]: r for r in read_csv_dicts(ns3_root / f"{out_prefix}_mac.csv")}

    return {
        "telemetry_pdr_pct": 100.0 * rx / possible if possible else 0.0,
        "telemetry_delay_ms": weighted(tel, "rx_receptions", "mean_delay_ms"),
        "telemetry_hops": weighted(tel, "rx_receptions", "mean_hops"),
        "data_pdr_pct": 100.0 * drx / doff if doff else 0.0,
        "data_delay_ms": weighted(dat, "rx_pkts", "mean_delay_ms"),
        "data_hops": weighted(dat, "rx_pkts", "mean_hops"),
        "data_throughput_kbps": sum(float(r["throughput_kbps"]) for r in dat),
        "mac_fail_L": int(mac.get("L", {}).get("failed_attempts", 0)),
        "mac_fail_Ku": int(mac.get("Ku", {}).get("failed_attempts", 0)),
        "mac_drop_L": int(mac.get("L", {}).get("frames_dropped_after_retries", 0)),
        "mac_drop_Ku": int(mac.get("Ku", {}).get("frames_dropped_after_retries", 0)),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--n", type=int, required=True, help="number of replications")
    ap.add_argument("--seed", type=int, default=None,
                    help="base seed; replication i uses Python seed (seed+i) and ns-3 "
                         "--run (seed+i). Omit for a fresh, non-reproducible seed each run "
                         "(ns-3 --run still varies 1,2,3,... across replications either way).")
    ap.add_argument("--work-dir", default="replications",
                    help="per-replication export directories go under here")
    ap.add_argument("--out-prefix", default="rep",
                    help="ns-3 result files are <out-prefix><i>_*.csv in --ns3-root, i=0..n-1")

    nb = ap.add_argument_group("notebook stage (same meaning as run_pipeline.py)")
    nb.add_argument("--sim-time", type=float, default=60)
    nb.add_argument("--calc-interval", type=float, default=5)
    nb.add_argument("--n-ships", type=int, default=5)
    nb.add_argument("--placement-budget-s", type=float, default=10.0)
    nb.add_argument("--dem-file", default=None)
    nb.add_argument("--notebook-file", default=None)
    nb.add_argument("--plots", action="store_true",
                    help="keep per-replication plots (off by default - see make_plots "
                         "in losGenerationBands.py's docstring for why)")

    ns = ap.add_argument_group("ns-3 stage (same meaning as run_pipeline.py)")
    ns.add_argument("--ns3-root", required=True)
    ns.add_argument("--cc-file", default=None)

    args, extra_ns3_args = ap.parse_known_args()
    if args.n < 1:
        ap.error("--n must be at least 1")

    here = Path(__file__).resolve().parent
    notebook_file = Path(args.notebook_file) if args.notebook_file else here / "losGenerationBands.py"
    cc_file = Path(args.cc_file) if args.cc_file else here / "uav_mesh_eval.cc"
    ns3_root = Path(args.ns3_root).resolve()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if not notebook_file.exists():
        ap.error(f"notebook file not found: {notebook_file}")
    if not cc_file.exists():
        ap.error(f"uav_mesh_eval.cc not found: {cc_file}")

    sys.path.insert(0, str(notebook_file.parent))
    nb_module = importlib.import_module(notebook_file.stem)
    scratch_dest = run_pipeline.sync_scratch_file(cc_file, ns3_root)
    scratch_name = scratch_dest.stem

    import numpy as np

    rows = []
    for i in range(args.n):
        rep_seed = (args.seed + i) if args.seed is not None else None
        if rep_seed is not None:
            np.random.seed(rep_seed)
        ns3_run = (args.seed if args.seed is not None else 1) + i

        export_dir = work_dir / "exports" / f"rep{i}"
        out_prefix_i = f"{args.out_prefix}{i}"

        print(f"\n[replications] ===== run {i + 1}/{args.n} "
             f"(seed={rep_seed}, ns-3 --run={ns3_run}) =====")
        nb_module.main(
            sim_time=args.sim_time,
            calc_interval=args.calc_interval,
            n_ships=args.n_ships,
            placement_budget_s=args.placement_budget_s,
            dem_file_path=args.dem_file,
            export_dir=str(export_dir),
            plot_dir=str(work_dir / "plots" / f"rep{i}"),
            make_plots=args.plots,
        )
        run_pipeline.run_ns3(ns3_root, scratch_name, export_dir, out_prefix_i,
                            [f"--run={ns3_run}"] + extra_ns3_args)

        try:
            summary = summarise_replication(ns3_root, out_prefix_i)
        except FileNotFoundError as e:
            print(f"[replications] WARNING: replication {i} produced no result files "
                 f"({e}) - ns-3 may have aborted. Skipping it in the aggregate.")
            continue
        summary["replication"] = i
        summary["seed"] = rep_seed
        rows.append(summary)
        print(f"[replications] run {i}: telemetry PDR {summary['telemetry_pdr_pct']:.2f}%, "
             f"data-transfer PDR {summary['data_pdr_pct']:.2f}%")

    if not rows:
        print("\n[replications] no successful replications - nothing to aggregate")
        return

    # ---- per-replication table -------------------------------------------
    field_order = ["replication", "seed", "telemetry_pdr_pct", "telemetry_delay_ms",
                  "telemetry_hops", "data_pdr_pct", "data_delay_ms", "data_hops",
                  "data_throughput_kbps", "mac_fail_L", "mac_fail_Ku", "mac_drop_L",
                  "mac_drop_Ku"]
    results_path = work_dir / "replication_results.csv"
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        w.writerows(rows)

    # ---- aggregate summary -------------------------------------------------
    metric_keys = [k for k in field_order if k not in ("replication", "seed")]
    summary_path = work_dir / "replication_summary.csv"
    print(f"\n===== AGGREGATE OVER {len(rows)} REPLICATIONS =====")
    print(f"{'metric':24s} {'mean':>12s} {'std':>12s} {'min':>12s} {'max':>12s}")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "mean", "std", "min", "max"])
        for key in metric_keys:
            vals = [r[key] for r in rows]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            w.writerow([key, len(vals), mean, std, min(vals), max(vals)])
            print(f"{key:24s} {mean:12.3f} {std:12.3f} {min(vals):12.3f} {max(vals):12.3f}")

    print(f"\n[replications] per-replication rows: {results_path}")
    print(f"[replications] aggregate summary:     {summary_path}")
    if len(rows) < args.n:
        print(f"[replications] NOTE: {args.n - len(rows)} replication(s) failed and were "
             f"excluded from the aggregate - see WARNINGs above")


if __name__ == "__main__":
    main()