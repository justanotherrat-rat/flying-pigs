#!/usr/bin/env python3
"""Adds the brute-force 'grid' placement mode to simulation/losGenerationBands.py and
simulation/run_replications.py.  Run from the repo root:  python3 apply_grid_mode.py

Replaces grid_mode.patch (which patch tools reject if whitespace is altered in transit).
Idempotent: running it twice does nothing the second time.  Each edit is anchored on exact
text; if one anchor is missing (you changed that code) it stops BEFORE writing anything and
tells you which one, so no file is ever left half-edited.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LGB = ROOT / "simulation" / "losGenerationBands.py"
REP = ROOT / "simulation" / "run_replications.py"

EDITS = {
    LGB: [
        ("main() signature",
         "         min_alt_m=1500.0, max_alt_m=6100.0):\n\n    if not np.isfinite(uav_speed_kmh)",
         "         min_alt_m=1500.0, max_alt_m=6100.0, grid_shape=(9, 9, 5)):\n\n    if not np.isfinite(uav_speed_kmh)"),
        ("main() mode validation",
         "    if placement_mode not in ('optimized', 'midpoint'):\n"
         "        raise ValueError(\"placement_mode must be 'optimized' or 'midpoint'\")\n",
         "    if placement_mode not in ('optimized', 'midpoint', 'grid'):\n"
         "        raise ValueError(\"placement_mode must be 'optimized', 'midpoint' or 'grid'\")\n"),
        ("main() place_uavs call",
         "                optimize=placement_mode == 'optimized',\n            )\n        except TimeoutError:",
         "                optimize=placement_mode == 'optimized',\n"
         "                grid_shape=grid_shape if placement_mode == 'grid' else None,\n"
         "            )\n        except TimeoutError:"),
        ("place_uavs() signature + docstring",
         "               uav_speed_kmh=130.0, optimize=True):\n"
         "    \"\"\"Place one UAV, optionally evaluating only the midpoint candidate.\"\"\"\n",
         "               uav_speed_kmh=130.0, optimize=True, grid_shape=None):\n"
         "    \"\"\"Place one UAV.\n\n"
         "    Default: midpoint seed + differential evolution within time_budget_s.\n"
         "    optimize=False: evaluate only the midpoint candidate.\n"
         "    grid_shape=(nx, ny, nz): brute-force reference - evaluate the midpoint seed plus\n"
         "    every point of an nx*ny*nz grid over the same search bounds through the SAME\n"
         "    objective (including the travel penalty) used by the optimiser; no time limit,\n"
         "    DE is not run (pass optimize=False with it).\n"
         "    \"\"\"\n"),
        ("place_uavs() search",
         "        evaluate(midpoint_seed)\n        if optimize:\n            differential_evolution(",
         "        evaluate(midpoint_seed)\n"
         "        if grid_shape is not None:\n"
         "            if len(grid_shape) != 3 or min(grid_shape) < 2:\n"
         "                raise ValueError('grid_shape must be three integers >= 2')\n"
         "            axes = [np.linspace(lo, hi, int(n))\n"
         "                    for (lo, hi), n in zip(bounds, grid_shape)]\n"
         "            for east in axes[0]:\n"
         "                for north in axes[1]:\n"
         "                    for altitude in axes[2]:\n"
         "                        evaluate(np.array([east, north, altitude]))\n"
         "        elif optimize:\n"
         "            differential_evolution("),
    ],
    REP: [
        ("--placement-mode option",
         "    ap.add_argument(\"--placement-mode\", choices=(\"optimized\", \"midpoint\"),\n"
         "                    default=\"optimized\",\n"
         "                    help=\"single placement mode to run (default: optimized)\")",
         "    ap.add_argument(\"--placement-mode\", choices=(\"optimized\", \"midpoint\", \"grid\"),\n"
         "                    default=\"optimized\",\n"
         "                    help=\"single placement mode to run (default: optimized). 'grid' is the \"\n"
         "                         \"brute-force reference (see --grid)\")\n"
         "    ap.add_argument(\"--grid\", type=int, nargs=3, default=(9, 9, 5), metavar=(\"NX\", \"NY\", \"NZ\"),\n"
         "                    help=\"grid resolution for --placement-mode grid (default 9 9 5)\")"),
        ("main() call",
         "                max_alt_m=args.uav_max_alt_m,\n            )\n            run_pipeline.run_ns3",
         "                max_alt_m=args.uav_max_alt_m,\n"
         "                grid_shape=tuple(args.grid),\n"
         "            )\n            run_pipeline.run_ns3"),
    ],
}


def main():
    new_text, problems, notes, crlf = {}, [], [], {}
    for path, edits in EDITS.items():
        if not path.is_file():
            sys.exit(f"not found: {path}  (run from the repo root)")
        raw = path.read_bytes().decode()
        crlf[path] = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        for name, old, new in edits:
            if new in text:
                notes.append(f"  already applied : {path.name}: {name}")
            elif text.count(old) == 1:
                text = text.replace(old, new)
                notes.append(f"  will apply      : {path.name}: {name}")
            else:
                problems.append(f"{path.name}: {name}  (anchor found {text.count(old)}x, expected 1)")
        new_text[path] = text
    print("\n".join(notes))
    if problems:
        print("\nNOTHING WAS WRITTEN. These anchors don't match your current code:", file=sys.stderr)
        print("\n".join("  - " + p for p in problems), file=sys.stderr)
        print("Send me those two files (or the named functions) and I'll adapt the edit.", file=sys.stderr)
        sys.exit(1)
    for path, text in new_text.items():          # keep each file's original line endings
        path.write_bytes((text.replace("\n", "\r\n") if crlf[path] else text).encode())
    print("\nDone. Check with: git diff --stat")


if __name__ == "__main__":
    main()