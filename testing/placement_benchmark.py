#!/usr/bin/env python3
"""Proxy-score benchmark of UAV placement methods over many random scenarios.

Produces the raw data for
  Table 2: brute-force grid vs DE (1 s / 10 s / 60 s)
  Table 4: midpoint placement vs DE (1 s / 10 s / 60 s)

For scenario i (i = 0..N-1) it seeds numpy with SEED+i, generates the ship convoy exactly
as losGenerationBands.main() does (same generator, same DEM, same ENU reference = ship 0),
so scenario i has the SAME ships as `run_replications.py --seed SEED` replication i at
step 0.  Every method then places the UAV for that one snapshot by calling the pipeline's
own place_uavs() (same objective, no travel constraint because there is no previous
position), so the methods differ only in the search strategy.

Methods: midpoint | grid | de<seconds>   (default: midpoint grid de1 de10 de60)

Run from the repo root, on an idle machine, one process at a time (DE budgets are
wall-clock, so load changes how many evaluations DE gets; the evaluation count is logged):

    python testing/placement_benchmark.py --n 50 --seed 1000 --out results/placement_benchmark.csv

Score (identical to place_uavs): 100000*disconnected_pairs + 10000*missing_two_hop_pairs
                                 + total UAV link length in km   (lower is better)
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import shortest_path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "simulation"))
import losGenerationBands as lgb  # noqa: E402


class EvalCounter:
    """Counts calls to the module's graph builder = placement objective evaluations."""

    def __init__(self, module):
        self.module = module
        self.original = module.build_dual_band_graph
        self.count = 0

    def __enter__(self):
        def counted(*a, **k):
            self.count += 1
            return self.original(*a, **k)
        self.module.build_dual_band_graph = counted
        return self

    def __exit__(self, *exc):
        self.module.build_dual_band_graph = self.original


def score_of(coords, n_ships, dem, lat_ref, lon_ref):
    """Re-derive place_uavs' objective for the returned position (no travel term)."""
    gl, gk = lgb.build_dual_band_graph(coords, dem, lat_ref, lon_ref, n_ships=n_ships)
    graph = gl.maximum(gk)
    adj = graph.toarray()
    m = lgb._network_metrics(adj)
    missing = m["total_pairs"] - m["pairs_within_two_hops"]
    dist = shortest_path(graph, directed=False, unweighted=True)
    pair = dist[np.triu_indices(len(coords), k=1)]
    disconnected = int(np.count_nonzero(~np.isfinite(pair)))
    link_km = float(np.sum(adj[-1, :n_ships]) / 1000.0)
    return (100000.0 * disconnected + 10000.0 * missing + link_km,
            link_km, disconnected, int(missing))


def make_scenario(dem, seed, n_ships):
    np.random.seed(seed)
    ships_lla = lgb.generate_convoy_lla(dem, n_ships=n_ships)
    lat_ref, lon_ref = ships_lla[0, 0], ships_lla[0, 1]
    ships_enu = lgb.lla_to_enu(ships_lla[:, 0], ships_lla[:, 1], ships_lla[:, 2],
                               lat_ref, lon_ref, 0.0).T
    return ships_enu, lat_ref, lon_ref


def run_method(method, ships, dem, lat_ref, lon_ref, args):
    kw = dict(min_alt_m=args.min_alt, max_alt_m=args.max_alt)
    if method == "midpoint":
        kw.update(optimize=False)
    elif method == "grid":
        kw.update(optimize=False, grid_shape=tuple(args.grid))
    elif method.startswith("de"):
        kw.update(optimize=True, time_budget_s=float(method[2:]), random_state=args.de_seed)
    else:
        raise ValueError(f"unknown method {method!r}")
    with EvalCounter(lgb) as ctr:
        t0 = time.perf_counter()
        res = lgb.place_uavs(ships, dem, lat_ref, lon_ref, **kw)
        elapsed = time.perf_counter() - t0
        evals = ctr.count
    pos = np.asarray(res["uavs"][0]["uav_enu"], dtype=float)
    score, link_km, disc, missing = score_of(np.vstack((ships, pos)), len(ships),
                                             dem, lat_ref, lon_ref)
    return pos, score, link_km, disc, missing, elapsed, evals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="number of scenarios")
    ap.add_argument("--seed", type=int, default=1000, help="scenario i uses seed+i")
    ap.add_argument("--n-ships", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=["midpoint", "grid", "de1", "de10", "de60"])
    ap.add_argument("--grid", type=int, nargs=3, default=[9, 9, 5], metavar=("NX", "NY", "NZ"))
    ap.add_argument("--de-seed", type=int, default=7,
                    help="DE random_state (pipeline default 7)")
    ap.add_argument("--min-alt", type=float, default=1500.0)
    ap.add_argument("--max-alt", type=float, default=6100.0)
    ap.add_argument("--dem", default=None)
    ap.add_argument("--out", default=str(REPO / "results" / "placement_benchmark.csv"))
    args = ap.parse_args()

    dem = lgb.load_dem(args.dem)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "seed", "method", "score", "uav_link_km", "disconnected_pairs",
              "missing_two_hop_pairs", "east_m", "north_m", "alt_m", "d_east_from_centroid_m",
              "d_north_from_centroid_m", "compute_s", "evaluations"]
    rows = []
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(args.n):
            seed = args.seed + i
            ships, lat_ref, lon_ref = make_scenario(dem, seed, args.n_ships)
            centroid = ships[:, :2].mean(axis=0)
            for m in args.methods:
                pos, score, link_km, disc, missing, el, ev = run_method(
                    m, ships, dem, lat_ref, lon_ref, args)
                row = dict(scenario=i, seed=seed, method=m, score=score, uav_link_km=link_km,
                           disconnected_pairs=disc, missing_two_hop_pairs=missing,
                           east_m=pos[0], north_m=pos[1], alt_m=pos[2],
                           d_east_from_centroid_m=pos[0] - centroid[0],
                           d_north_from_centroid_m=pos[1] - centroid[1],
                           compute_s=el, evaluations=ev)
                w.writerow(row)
                f.flush()
                rows.append(row)
                print(f"[{i + 1}/{args.n}] seed={seed} {m:>8}: score={score:12.2f} "
                      f"link={link_km:7.2f} km  {el:6.2f}s  {ev} evals", flush=True)

    print(f"\nWrote {out}\n\nMean over {args.n} scenarios:")
    print(f"{'method':>9} {'score':>12} {'link_km':>8} {'dE_km':>7} {'dN_km':>7} "
          f"{'alt_m':>7} {'time_s':>8} {'evals':>7}")
    for m in args.methods:
        r = [x for x in rows if x["method"] == m]
        mean = lambda k: statistics.mean(x[k] for x in r)
        print(f"{m:>9} {mean('score'):12.2f} {mean('uav_link_km'):8.2f} "
              f"{mean('d_east_from_centroid_m') / 1000:7.2f} "
              f"{mean('d_north_from_centroid_m') / 1000:7.2f} {mean('alt_m'):7.0f} "
              f"{mean('compute_s'):8.2f} {mean('evaluations'):7.0f}")


if __name__ == "__main__":
    main()