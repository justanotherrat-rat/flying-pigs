#!/usr/bin/env python3
"""
run_pipeline.py -- one command for "notebook export -> ns-3 run".

WHAT THIS DOES
--------------
1. Imports losGenerationBands.py and calls its main(...) directly (in-process,
   not a subprocess) to run the ship+UAV placement simulation and write
   nodes.csv/links.csv into an export directory.
2. Copies uav_mesh_eval.cc into <ns3-root>/scratch/ (only if it changed, so
   ns-3 doesn't needlessly rebuild every run).
3. Runs `./ns3 run "scratch/uav_mesh_eval --dir=<export dir> --out=<out> ..."`
   in <ns3-root>, streaming its output live, so you see the same console
   output (link budget, per-step link counts, final summary) you'd get
   running it by hand.

This does NOT merge the two into a single binary - ns-3 is still a separate
compiled C++ program, and Python can't be that program. What this gives you
is a single command that does both steps in the right order with the right
paths, instead of you doing it by hand and re-hitting the "--dir=... doesn't
exist" class of mistake that came up repeatedly while developing this by hand.

REQUIREMENTS
------------
- losGenerationBands.py (and its dependencies: numpy, scipy, rasterio, pyproj,
  matplotlib, networkx) importable from wherever you run this script - i.e.
  the same Python environment you already run the notebook in. Simplest: put
  this file in the same directory as losGenerationBands.py.
- A working ns-3 checkout (the same one you've been running `./ns3 run`
  in by hand).

USAGE
-----
  # one run, with the UAV
  python3 run_pipeline.py --ns3-root ~/ns-3-dev --out with_uav --useUav=1

  # one run, ships only (baseline)
  python3 run_pipeline.py --ns3-root ~/ns-3-dev --out no_uav --useUav=0

  # both in one command (reuses the SAME export - see --compare's docstring
  # below for why that matters)
  python3 run_pipeline.py --ns3-root ~/ns-3-dev --out result --compare

  # anything not recognised by this script (--seaState, --fadeKssDb,
  # --routing, --telemetryPeriodS, ...) is forwarded to uav_mesh_eval.cc
  # UNCHANGED, so every flag it supports works here too:
  python3 run_pipeline.py --ns3-root ~/ns-3-dev --out stress \\
      --seaState=6 --fadeKssDb=-100 --routing=aodv

Run with --help to see this script's own flags; run the .cc file's --help
(via `./ns3 run "scratch/uav_mesh_eval --PrintHelp"` in ns3-root) for the
full list of simulation flags this script forwards untouched.
"""
import argparse
import filecmp
import math
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM_OUT = PROJECT_ROOT / "sim-out"


def sync_scratch_file(cc_file: Path, ns3_root: Path) -> Path:
    """Copy cc_file into <ns3_root>/scratch/, skipping the copy if the file
    there is already identical (avoids bumping its mtime and triggering an
    unnecessary rebuild on every single run)."""
    scratch_dir = ns3_root / "scratch"
    if not scratch_dir.is_dir():
        raise FileNotFoundError(
            f"{scratch_dir} does not exist - is --ns3-root really an ns-3 checkout?")
    dest = scratch_dir / cc_file.name
    if dest.exists() and filecmp.cmp(cc_file, dest, shallow=False):
        print(f"[pipeline] {dest} already up to date, not copying")
    else:
        shutil.copy2(cc_file, dest)
        print(f"[pipeline] copied {cc_file} -> {dest}")
    return dest


def run_ns3(ns3_root: Path, scratch_name: str, export_dir: Path, out_prefix: str,
            extra_args: list[str]) -> None:
    """Runs `./ns3 run "scratch/<scratch_name> --dir=... --out=... <extra>"`
    in ns3_root, with output streamed live (not captured), and raises on a
    nonzero exit so a failed sim run stops the pipeline instead of silently
    continuing to a summary step that has nothing to summarise."""
    ns3_exe = ns3_root / "ns3"
    if not ns3_exe.exists():
        raise FileNotFoundError(f"{ns3_exe} not found - is --ns3-root really an ns-3 checkout?")

    # export_dir is always made absolute before being handed to ns-3: `./ns3
    # run` executes with ns3_root as the working directory, so a relative
    # --dir here would be resolved against ns3_root, not against wherever
    # this script was invoked from - the exact "--dir=ns3_export: No such
    # file" mistake that came up more than once running this by hand.
    dir_abs = export_dir.resolve()
    output_prefix = Path(out_prefix).expanduser()
    if not output_prefix.is_absolute():
        output_prefix = SIM_OUT / "ns3" / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    run_arg = f"scratch/{scratch_name} --dir={dir_abs} --out={output_prefix}"
    if extra_args:
        run_arg += " " + " ".join(extra_args)

    print(f"\n[pipeline] running in {ns3_root}:\n  ./ns3 run \"{run_arg}\"\n")
    subprocess.run(["./ns3", "run", run_arg], cwd=ns3_root, check=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- notebook stage ------------------------------------------------
    nb = ap.add_argument_group("notebook (losGenerationBands.py) stage")
    nb.add_argument("--sim-time", type=float, default=60, help="minutes")
    nb.add_argument("--calc-interval", type=float, default=5, help="minutes between steps")
    nb.add_argument("--n-ships", type=int, default=5)
    nb.add_argument("--placement-budget-s", type=float, default=10.0)
    nb.add_argument("--uav-speed-kmh", type=float, default=130.0,
                    help="UAV cruise speed between placements (default 130 km/h; "
                         "typical range 100-160)")
    nb.add_argument("--uav-min-alt-m", type=float, default=1500.0,
                    help="minimum UAV altitude in metres (default 1500)")
    nb.add_argument("--uav-max-alt-m", type=float, default=6100.0,
                    help="maximum UAV altitude in metres (default 6100)")
    nb.add_argument("--dem-file", default=None, help="path to DEM GeoTIFF")
    nb.add_argument("--export-dir", default=str(SIM_OUT / "exports" / "pipeline"),
                    help="where nodes.csv/links.csv are written and read from")
    nb.add_argument("--plot-dir", default=str(SIM_OUT / "plots"))
    nb.add_argument("--skip-notebook", action="store_true",
                    help="reuse an existing --export-dir instead of re-running the "
                         "notebook stage (e.g. when only re-running ns-3 with "
                         "different radio/traffic flags on the same scenario)")
    nb.add_argument("--notebook-file", default=None,
                    help="path to losGenerationBands.py; default looks next to this script")

    # --- ns-3 stage ------------------------------------------------------
    ns = ap.add_argument_group("ns-3 (uav_mesh_eval.cc) stage")
    ns.add_argument("--ns3-root", required=True, help="path to your ns-3 checkout")
    ns.add_argument("--cc-file", default=None,
                    help="path to uav_mesh_eval.cc; default looks next to this script")
    ns.add_argument("--out", default=str(SIM_OUT / "ns3" / "result"),
                    help="ns-3 output file prefix (relative paths go under sim-out/ns3)")
    ns.add_argument("--compare", action="store_true",
                    help="run BOTH --useUav=1 (as <out>_uav) and --useUav=0 (as "
                         "<out>_baseline) against the SAME export, so the two are a fair "
                         "comparison instead of two different random placements")

    args, extra_ns3_args = ap.parse_known_args()
    # Anything this script doesn't recognise (--seaState=6, --routing=aodv,
    # --fadeKssDb=-100, ...) is forwarded to uav_mesh_eval.cc as-is. This is
    # deliberate rather than duplicating its whole flag list here: those
    # flags are defined in uav_mesh_eval.cc and will drift out of sync with
    # any copy kept here.
    if any(a in ("--useUav", "--useUav=0", "--useUav=1", "--useUav=true", "--useUav=false")
          for a in extra_ns3_args) and args.compare:
        ap.error("--compare sets --useUav itself for both runs; don't pass --useUav "
                 "alongside --compare")
    if (not math.isfinite(args.uav_min_alt_m) or
            not math.isfinite(args.uav_max_alt_m) or
            args.uav_min_alt_m < 0 or args.uav_min_alt_m > args.uav_max_alt_m):
        ap.error("UAV altitude range must be finite, non-negative, and ordered")

    here = Path(__file__).resolve().parent
    notebook_file = Path(args.notebook_file) if args.notebook_file else here / "losGenerationBands.py"
    cc_file = Path(args.cc_file) if args.cc_file else here / "uav_mesh_eval.cc"
    ns3_root = Path(args.ns3_root).resolve()
    export_dir = Path(args.export_dir)

    if not notebook_file.exists():
        ap.error(f"notebook file not found: {notebook_file} (pass --notebook-file)")
    if not cc_file.exists():
        ap.error(f"uav_mesh_eval.cc not found: {cc_file} (pass --cc-file)")

    # ---- stage 1: notebook -----------------------------------------------
    if args.skip_notebook:
        if not (export_dir / "nodes.csv").exists():
            ap.error(f"--skip-notebook given but {export_dir}/nodes.csv doesn't exist")
        print(f"[pipeline] --skip-notebook: reusing existing {export_dir}/")
    else:
        print(f"[pipeline] running notebook stage from {notebook_file}")
        sys.path.insert(0, str(notebook_file.parent))
        import importlib
        nb_module = importlib.import_module(notebook_file.stem)
        nb_module.main(
            sim_time=args.sim_time,
            calc_interval=args.calc_interval,
            n_ships=args.n_ships,
            placement_budget_s=args.placement_budget_s,
            uav_speed_kmh=args.uav_speed_kmh,
            min_alt_m=args.uav_min_alt_m,
            max_alt_m=args.uav_max_alt_m,
            dem_file_path=args.dem_file,
            export_dir=str(export_dir),
            plot_dir=args.plot_dir,
        )

    # ---- stage 2: ns-3 ----------------------------------------------------
    scratch_dest = sync_scratch_file(cc_file, ns3_root)
    scratch_name = scratch_dest.stem  # "uav_mesh_eval"

    if args.compare:
        run_ns3(ns3_root, scratch_name, export_dir, f"{args.out}_uav",
               ["--useUav=1"] + extra_ns3_args)
        run_ns3(ns3_root, scratch_name, export_dir, f"{args.out}_baseline",
               ["--useUav=0"] + extra_ns3_args)
        print(f"\n[pipeline] done. Compare {args.out}_uav_*.csv against "
             f"{args.out}_baseline_*.csv in {ns3_root}")
    else:
        run_ns3(ns3_root, scratch_name, export_dir, args.out, extra_ns3_args)
        print(f"\n[pipeline] done. Results: {args.out}_*.csv in {ns3_root}")


if __name__ == "__main__":
    main()