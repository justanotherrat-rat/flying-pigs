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
  4. parse that replication's FlowMonitor and link-budget outputs, and record
      per-ship threshold checks alongside one summary row.
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

Results land in <work-dir>/replication_results.csv (one row per replication),
<work-dir>/replication_ship_metrics.csv (one row per ship per replication),
and <work-dir>/replication_summary.csv (aggregated by placement when comparing).
By default, <work-dir> is the repository's sim-out/ directory; ns-3 CSV outputs
are stored in its ns3/ subdirectory.
Use --compare-midpoint to compare the optimizer with the midpoint-only baseline
on paired ship scenarios, or --placement-mode midpoint to run only the baseline.
Signal and ETX are explicitly link-budget estimates; latency, throughput, and
PDR are measured from ns-3's ship-to-UAV application flows.
"""
import argparse
import csv
import importlib
import math
import statistics
import sys
from pathlib import Path

# run_pipeline.py must be importable (same directory, by default)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline  # noqa: E402 - needs the sys.path insert above first

SIGNAL_MIN_DBM = -70.0
LATENCY_MAX_MS = 20.0
THROUGHPUT_MIN_KBPS = 10.0
ETX_MAX = 2.2
PDR_MIN_PCT = 99.0


def read_csv_dicts(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def route_link_metrics(linkbudget_path: Path, nodes_path: Path, ship_ids: list[int],
                       uav_node: int) -> dict[int, dict[str, float]]:
    """Return each ship's worst modeled direct-to-UAV RSSI and ETX over all time steps.

    Each edge ETX is estimated as 1 / (1 - outage probability), using ns-3's
    link-budget outage estimate. A ship without a direct ship-to-UAV edge at
    every step receives failing signal and ETX values. These use mean received
    power and are not packet-observed metrics.
    """
    steps = sorted({int(row["step"]) for row in read_csv_dicts(nodes_path)})
    direct_by_step: dict[int, dict[int, tuple[float, float]]] = {}
    for row in read_csv_dicts(linkbudget_path):
        source = int(row["i"])
        target = int(row["j"])
        if source != uav_node and target != uav_node:
            continue
        ship = target if source == uav_node else source
        if ship not in ship_ids:
            continue
        outage = float(row["outage_prob"])
        if outage >= 1.0:
            continue
        edge_etx = 1.0 / (1.0 - outage)
        direct_by_step.setdefault(int(row["step"]), {})[ship] = (
            edge_etx,
            float(row["rx_dbm"]),
        )

    result = {}
    for ship in ship_ids:
        step_metrics = []
        for step in steps:
            direct_link = direct_by_step.get(step, {}).get(ship)
            if direct_link is not None:
                edge_etx, rx_dbm = direct_link
                step_metrics.append((rx_dbm, edge_etx))

        if len(step_metrics) == len(steps) and steps:
            result[ship] = {
                "min_model_path_rx_dbm": min(item[0] for item in step_metrics),
                "max_model_path_etx": max(item[1] for item in step_metrics),
            }
        else:
            result[ship] = {
                "min_model_path_rx_dbm": -math.inf,
                "max_model_path_etx": math.inf,
            }
    return result


def evaluate_ship_metrics(data_rows: list[dict], link_metrics: dict[int, dict[str, float]],
                          replication: int) -> list[dict]:
    """Combine measured ship-to-UAV app results with modeled route metrics."""
    evaluated = []
    for row in data_rows:
        ship = int(row["ship"])
        route = link_metrics[ship]
        pdr_pct = 100.0 * float(row["pdr"])
        if "offered_pkts" in row and int(row["offered_pkts"]):
            pdr_pct = 100.0 * int(row["rx_pkts"]) / int(row["offered_pkts"])
        latency_ms = float(row["mean_delay_ms"])
        throughput_kbps = float(row["throughput_kbps"])
        checks = {
            "model_signal_pass": route["min_model_path_rx_dbm"] >= SIGNAL_MIN_DBM,
            "latency_pass": latency_ms < LATENCY_MAX_MS,
            "throughput_pass": throughput_kbps > THROUGHPUT_MIN_KBPS,
            "model_etx_pass": route["max_model_path_etx"] < ETX_MAX,
            "pdr_pass": pdr_pct > PDR_MIN_PCT,
        }
        evaluated.append({
            "replication": replication,
            "ship": ship,
            "min_model_path_rx_dbm": route["min_model_path_rx_dbm"],
            "max_model_path_etx": route["max_model_path_etx"],
            "latency_ms": latency_ms,
            "throughput_kbps": throughput_kbps,
            "pdr_pct": pdr_pct,
            **{key: int(value) for key, value in checks.items()},
            "all_criteria_pass": int(all(checks.values())),
        })
    return evaluated


def summarise_replication(ns3_output_dir: Path, out_prefix: str, export_dir: Path,
                          replication: int, ship_count: int) -> tuple[dict, list[dict]]:
    """Reads one replication's *_telemetry.csv / *_datatransfer.csv /
    *_mac.csv (written by uav_mesh_eval.cc into ns3_output_dir) and reduces each to
    the handful of scalars worth tracking across replications. Per-ship rows
    are combined by summing counts (so the result is a true fleet-wide PDR,
    not an average of five PDR percentages) and weighting delay/hop averages
    by each ship's received-packet count."""

    def weighted(rows, rx_key, val_key):
        totalW = sum(int(r[rx_key]) for r in rows)
        if totalW == 0:
            return 0.0
        return sum(int(r[rx_key]) * float(r[val_key]) for r in rows) / totalW

    tel = read_csv_dicts(ns3_output_dir / f"{out_prefix}_telemetry.csv")
    rx = sum(int(r["rx_receptions"]) for r in tel)
    possible = sum(int(r["possible_receptions"]) for r in tel)

    dat = read_csv_dicts(ns3_output_dir / f"{out_prefix}_datatransfer.csv")
    drx = sum(int(r["rx_pkts"]) for r in dat)
    doff = sum(int(r["offered_pkts"]) for r in dat)
    uav_node = ship_count
    link_metrics = route_link_metrics(
        ns3_output_dir / f"{out_prefix}_linkbudget.csv",
        export_dir / "nodes.csv",
        list(range(ship_count)),
        uav_node,
    )
    ship_metrics = evaluate_ship_metrics(dat, link_metrics, replication)

    mac = {r["band"]: r for r in read_csv_dicts(ns3_output_dir / f"{out_prefix}_mac.csv")}

    summary = {
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
        "criteria_ship_count": ship_count,
    }
    for key in ("model_signal_pass", "latency_pass", "throughput_pass",
                "model_etx_pass", "pdr_pass", "all_criteria_pass"):
        summary[f"{key}_ships"] = sum(row[key] for row in ship_metrics)
    summary["all_criteria_pass_pct"] = (
        100.0 * summary["all_criteria_pass_ships"] / ship_count
        if ship_count else 0.0
    )
    return summary, ship_metrics


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--n", type=int, default = 10, help="number of replications")
    ap.add_argument("--seed", type=int, default=None,
                    help="base seed; replication i uses Python seed (seed+i) and ns-3 "
                         "--run (seed+i). Omit for a fresh, non-reproducible seed each run "
                         "(ns-3 --run still varies 1,2,3,... across replications either way).")
    project_root = Path(__file__).resolve().parents[1]
    default_work_dir = project_root / "sim-out"
    ap.add_argument("--work-dir", default=str(default_work_dir),
                    help="directory for exports, ns-3 outputs, plots, and summaries "
                         f"(default: {default_work_dir})")
    ap.add_argument("--out-prefix", default="rep",
                    help="ns-3 result files are <out-prefix><i>_*.csv in "
                         "<work-dir>/ns3, i=0..n-1")
    ap.add_argument("--placement-mode", choices=("optimized", "midpoint", "grid"),
                    default="optimized",
                    help="single placement mode to run (default: optimized). 'grid' is the "
                         "brute-force reference (see --grid)")
    ap.add_argument("--grid", type=int, nargs=3, default=(9, 9, 5), metavar=("NX", "NY", "NZ"),
                    help="grid resolution for --placement-mode grid (default 9 9 5)")
    ap.add_argument("--compare-midpoint", action="store_true",
                    help="run optimized and midpoint placements for each identical scenario")

    nb = ap.add_argument_group("notebook stage (same meaning as run_pipeline.py)")
    nb.add_argument("--sim-time", type=float, default=60)
    nb.add_argument("--calc-interval", type=float, default=5)
    nb.add_argument("--n-ships", type=int, default=5)
    nb.add_argument("--placement-budget-s", type=float, default=10.0)
    nb.add_argument("--uav-speed-kmh", type=float, default=130.0,
                    help="UAV cruise speed between placements (default 130 km/h; "
                         "typical range 100-160)")
    nb.add_argument("--uav-min-alt-m", type=float, default=1500.0,
                    help="minimum UAV altitude in metres (default 1500)")
    nb.add_argument("--uav-max-alt-m", type=float, default=6100.0,
                    help="maximum UAV altitude in metres (default 6100)")
    nb.add_argument("--dem-file", default=None)
    nb.add_argument("--notebook-file", default=None)
    nb.add_argument("--plots", action="store_true",
                    help="keep per-replication plots (off by default - see make_plots "
                         "in losGenerationBands.py's docstring for why)")

    ns = ap.add_argument_group("ns-3 stage (same meaning as run_pipeline.py)")
    ns.add_argument("--ns3-root", default = str(Path("~/ns-3-dev").expanduser().resolve()))
    ns.add_argument("--cc-file", default=None)

    args, extra_ns3_args = ap.parse_known_args()
    if args.n < 1:
        ap.error("--n must be at least 1")
    if (not math.isfinite(args.uav_min_alt_m) or
            not math.isfinite(args.uav_max_alt_m) or
            args.uav_min_alt_m < 0 or args.uav_min_alt_m > args.uav_max_alt_m):
        ap.error("UAV altitude range must be finite, non-negative, and ordered")

    here = Path(__file__).resolve().parent
    notebook_file = Path(args.notebook_file) if args.notebook_file else here / "losGenerationBands.py"
    cc_file = Path(args.cc_file) if args.cc_file else here / "uav_mesh_eval.cc"
    ns3_root = Path(args.ns3_root).resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    ns3_output_dir = work_dir / "ns3"
    ns3_output_dir.mkdir(parents=True, exist_ok=True)

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
    all_ship_metrics = []
    placements = ("optimized", "midpoint") if args.compare_midpoint else (args.placement_mode,)
    comparison_rows = []
    for i in range(args.n):
        rep_seed = (args.seed + i) if args.seed is not None else None
        if args.compare_midpoint and rep_seed is None:
            rep_seed = int(np.random.SeedSequence().generate_state(1)[0])
        ns3_run = (args.seed if args.seed is not None else 1) + i
        paired_summaries = {}

        for placement in placements:
            if rep_seed is not None:
                np.random.seed(rep_seed)
            export_dir = work_dir / "exports" / f"rep{i}"
            if args.compare_midpoint:
                export_dir /= placement
                out_prefix_i = f"{args.out_prefix}{i}_{placement}"
            else:
                out_prefix_i = f"{args.out_prefix}{i}"
            output_prefix = ns3_output_dir / out_prefix_i

            print(f"\n[replications] ===== run {i + 1}/{args.n} "
                  f"({placement}, seed={rep_seed}, ns-3 --run={ns3_run}) =====")
            nb_module.main(
                sim_time=args.sim_time,
                calc_interval=args.calc_interval,
                n_ships=args.n_ships,
                placement_budget_s=args.placement_budget_s,
                uav_speed_kmh=args.uav_speed_kmh,
                dem_file_path=args.dem_file,
                export_dir=str(export_dir),
                plot_dir=str(work_dir / "plots" / f"rep{i}" / placement),
                make_plots=args.plots,
                placement_mode=placement,
                min_alt_m=args.uav_min_alt_m,
                max_alt_m=args.uav_max_alt_m,
                grid_shape=tuple(args.grid),
            )
            run_pipeline.run_ns3(ns3_root, scratch_name, export_dir, str(output_prefix),
                                [f"--run={ns3_run}"] + extra_ns3_args)

            try:
                summary, ship_metrics = summarise_replication(
                    ns3_output_dir, out_prefix_i, export_dir, i, args.n_ships)
            except FileNotFoundError as e:
                print(f"[replications] WARNING: replication {i} ({placement}) "
                      f"produced no result files ({e}) - skipping this result.")
                continue
            summary["replication"] = i
            summary["seed"] = rep_seed
            summary["placement"] = placement
            rows.append(summary)
            all_ship_metrics.extend([{**row, "placement": placement} for row in ship_metrics])
            paired_summaries[placement] = summary
            print(f"[replications] run {i} {placement}: data-transfer PDR "
                  f"{summary['data_pdr_pct']:.2f}%, ships meeting all thresholds "
                  f"{summary['all_criteria_pass_ships']}/{summary['criteria_ship_count']}")

        if args.compare_midpoint and all(mode in paired_summaries for mode in placements):
            optimized = paired_summaries["optimized"]
            midpoint = paired_summaries["midpoint"]
            comparison = {"replication": i, "seed": rep_seed}
            for key in ("data_pdr_pct", "data_delay_ms", "data_throughput_kbps",
                        "all_criteria_pass_pct"):
                comparison[f"optimized_{key}"] = optimized[key]
                comparison[f"midpoint_{key}"] = midpoint[key]
                comparison[f"delta_{key}"] = optimized[key] - midpoint[key]
            comparison_rows.append(comparison)

    if not rows:
        print("\n[replications] no successful replications - nothing to aggregate")
        return

    # ---- per-replication table -------------------------------------------
    field_order = ["replication", "seed", "placement", "telemetry_pdr_pct", "telemetry_delay_ms",
                  "telemetry_hops", "data_pdr_pct", "data_delay_ms", "data_hops",
                  "data_throughput_kbps", "mac_fail_L", "mac_fail_Ku", "mac_drop_L",
                  "mac_drop_Ku", "criteria_ship_count", "model_signal_pass_ships",
                  "latency_pass_ships", "throughput_pass_ships", "model_etx_pass_ships",
                  "pdr_pass_ships", "all_criteria_pass_ships", "all_criteria_pass_pct"]
    results_path = work_dir / "replication_results.csv"
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        w.writerows(rows)

    ship_metrics_path = work_dir / "replication_ship_metrics.csv"
    ship_fields = ["replication", "placement", "ship", "min_model_path_rx_dbm", "max_model_path_etx",
                   "latency_ms", "throughput_kbps", "pdr_pct", "model_signal_pass",
                   "latency_pass", "throughput_pass", "model_etx_pass", "pdr_pass",
                   "all_criteria_pass"]
    with open(ship_metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ship_fields)
        w.writeheader()
        w.writerows(all_ship_metrics)

    # ---- aggregate summary -------------------------------------------------
    metric_keys = [k for k in field_order if k not in ("replication", "seed", "placement")]
    summary_path = work_dir / "replication_summary.csv"
    if args.compare_midpoint:
        print(f"\n===== AGGREGATE OVER {len(comparison_rows)} PAIRED SCENARIOS =====")
    else:
        print(f"\n===== AGGREGATE OVER {len(rows)} REPLICATIONS =====")
    print(f"{'metric':24s} {'mean':>12s} {'std':>12s} {'min':>12s} {'max':>12s}")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        if args.compare_midpoint:
            w.writerow(["placement", "metric", "n", "mean", "std", "min", "max"])
            for placement in placements:
                print(f"\n--- {placement} ---")
                placement_rows = [row for row in rows if row["placement"] == placement]
                if not placement_rows:
                    print("no successful runs")
                    continue
                for key in metric_keys:
                    vals = [r[key] for r in placement_rows]
                    mean = statistics.mean(vals)
                    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    w.writerow([placement, key, len(vals), mean, std, min(vals), max(vals)])
                    print(f"{key:24s} {mean:12.3f} {std:12.3f} {min(vals):12.3f} {max(vals):12.3f}")
        else:
            w.writerow(["metric", "n", "mean", "std", "min", "max"])
            for key in metric_keys:
                vals = [r[key] for r in rows]
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                w.writerow([key, len(vals), mean, std, min(vals), max(vals)])
                print(f"{key:24s} {mean:12.3f} {std:12.3f} {min(vals):12.3f} {max(vals):12.3f}")

    comparison_path = None
    comparison_summary_path = None
    if args.compare_midpoint:
        comparison_path = work_dir / "replication_comparison.csv"
        comparison_fields = ["replication", "seed"]
        delta_keys = []
        for key in ("data_pdr_pct", "data_delay_ms", "data_throughput_kbps",
                    "all_criteria_pass_pct"):
            comparison_fields.extend((f"optimized_{key}", f"midpoint_{key}", f"delta_{key}"))
            delta_keys.append(f"delta_{key}")
        with open(comparison_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=comparison_fields)
            w.writeheader()
            w.writerows(comparison_rows)
        comparison_summary_path = work_dir / "replication_comparison_summary.csv"
        with open(comparison_summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "n", "mean", "std", "min", "max"])
            for key in delta_keys:
                values = [row[key] for row in comparison_rows]
                if not values:
                    continue
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                w.writerow([key, len(values), mean, std, min(values), max(values)])

    print(f"\n[replications] per-replication rows: {results_path}")
    print(f"[replications] per-ship criteria:    {ship_metrics_path}")
    if comparison_path:
        print(f"[replications] paired comparison:    {comparison_path}")
        print(f"[replications] paired delta summary: {comparison_summary_path}")
    total_ship_scenarios = len(all_ship_metrics)
    total_joint_pass = sum(row["all_criteria_pass"] for row in all_ship_metrics)
    print(f"[replications] all-threshold pass:    {total_joint_pass}/"
          f"{total_ship_scenarios} ship-scenarios "
          f"({100.0 * total_joint_pass / total_ship_scenarios:.2f}%)")
    print(f"[replications] aggregate summary:     {summary_path}")
    if len(rows) < args.n:
        print(f"[replications] NOTE: {args.n - len(rows)} replication(s) failed and were "
             f"excluded from the aggregate - see WARNINGs above")


if __name__ == "__main__":
    main()