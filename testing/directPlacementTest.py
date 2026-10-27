"""Directly validate UAV placement against a grid.
Run from the repository root:
    python testing/directPlacementTest.py

By default this uses output_hh.tiff in the repository root. Use --dem 
to select a DEM and --module to test another placement implementation.
"""
import argparse
import importlib.util
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
from scipy.sparse.csgraph import shortest_path


class _Bounds:
    def __init__(self, left, bottom, right, top):
        self.left, self.bottom, self.right, self.top = left, bottom, right, top


class _RasterDEM:
    """Cache the WGS84 raster so graph evaluations reuse its loaded band."""

    def __init__(self, path):
        with rasterio.open(path) as dataset:
            if dataset.crs is None or dataset.crs.to_epsg() != 4326:
                raise ValueError(
                    f"DEM must use EPSG:4326 coordinates; got {dataset.crs}"
                )
            self.data = dataset.read(1)
            self.transform = dataset.transform
            self.height = dataset.height
            self.width = dataset.width
            self.nodata = dataset.nodata
            self.bounds = _Bounds(
                dataset.bounds.left,
                dataset.bounds.bottom,
                dataset.bounds.right,
                dataset.bounds.top,
            )

    def index(self, lon, lat):
        row, col = rowcol(self.transform, lon, lat)
        return int(row), int(col)

    def read(self, band=1, window=None):
        if window is None:
            return self.data
        row_start, col_start = int(window.row_off), int(window.col_off)
        row_end = row_start + int(window.height)
        col_end = col_start + int(window.width)
        return self.data[row_start:row_end, col_start:col_end]


def _load_module(module_path):
    module_path = Path(module_path).expanduser().resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"placement module not found: {module_path}")
    loader = SourceFileLoader("placement_under_test", str(module_path))
    spec = importlib.util.spec_from_loader("placement_under_test", loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load placement module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bounds(ships, min_alt, max_alt):
    horizontal_min = ships[:, :2].min(axis=0)
    horizontal_max = ships[:, :2].max(axis=0)
    margin = 0.15 * np.maximum(horizontal_max - horizontal_min, 1000.0)
    return (
        (horizontal_min[0] - margin[0], horizontal_max[0] + margin[0]),
        (horizontal_min[1] - margin[1], horizontal_max[1] + margin[1]),
        (min_alt, max_alt),
    )


def _score_position(module, ships, position, dem, lat_ref, lon_ref):
    coords = np.vstack((ships, np.asarray(position, dtype=float).reshape(1, 3)))
    graph_l, graph_ku = module.build_dual_band_graph(
        coords, dem, lat_ref, lon_ref, n_ships=len(ships)
    )
    graph = graph_l.maximum(graph_ku)
    adjacency = graph.toarray()
    metrics = module._network_metrics(adjacency)
    missing_two_hop = metrics["total_pairs"] - metrics["pairs_within_two_hops"]
    distances = shortest_path(graph, directed=False, unweighted=True)
    pair_distances = distances[np.triu_indices(len(coords), k=1)]
    disconnected = int(np.count_nonzero(~np.isfinite(pair_distances)))
    uav_link_length_km = np.sum(adjacency[-1, :len(ships)]) / 1000.0
    score = (
        100000.0 * disconnected
        + 10000.0 * missing_two_hop
        + uav_link_length_km
    )
    return float(score), float(uav_link_length_km)


def _grid_search(module, ships, dem, lat_ref, lon_ref, bounds, dimensions):
    axes = [np.linspace(low, high, count)
            for (low, high), count in zip(bounds, dimensions)]
    best_score = float("inf")
    best_position = None
    best_link_length_km = None
    started = time.perf_counter()
    for east in axes[0]:
        for north in axes[1]:
            for altitude in axes[2]:
                position = np.array([east, north, altitude])
                score, link_length_km = _score_position(
                    module, ships, position, dem, lat_ref, lon_ref
                )
                if score < best_score:
                    best_score, best_position = score, position
                    best_link_length_km = link_length_km
    return best_score, best_position, best_link_length_km, time.perf_counter() - started


def _within_bounds(position, bounds):
    return all(
        low - 1e-8 <= value <= high + 1e-8
        for value, (low, high) in zip(position, bounds)
    )


def _format_position(position):
    return (
        f"E={position[0] / 1000:.2f} km, "
        f"N={position[1] / 1000:.2f} km, Alt={position[2]:.1f} m"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        default=Path(__file__).resolve().parents[1]
        / "simulation"
        / "losGenerationBands",
        help="Python module under test (default: simulation/losGenerationBands)",
    )
    parser.add_argument(
        "--dem",
        type=Path,
        default=None,
        help="WGS84 GeoTIFF (default: simulation/output_hh.tif)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--budgets", type=float, nargs="+", default=[1.0, 10.0])
    parser.add_argument("--grid", type=int, nargs=3, default=[9, 9, 5],
                        metavar=("NX", "NY", "NZ"))
    parser.add_argument("--min-alt", type=float, default=1500.0)
    parser.add_argument("--max-alt", type=float, default=6100.0)
    parser.add_argument("--maxiter", type=int, default=35)
    parser.add_argument("--popsize", type=int, default=8)
    args = parser.parse_args()

    if not args.seeds:
        parser.error("provide at least one seed")
    if not args.budgets or any(budget < 0 for budget in args.budgets):
        parser.error("budgets must be non-negative")
    if any(points < 2 for points in args.grid):
        parser.error("each grid dimension must be at least 2")
    if args.min_alt > args.max_alt:
        parser.error("--min-alt must not exceed --max-alt")

    repository_root = Path(__file__).resolve().parents[1]
    module = _load_module(args.module)
    dem_path = args.dem
    if dem_path is None:
        dem_path = repository_root / "simulation" / "output_hh.tif"
    elif not dem_path.is_absolute():
        dem_path = (Path.cwd() / dem_path).resolve()
    if not dem_path.is_file():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    lat_ref, lon_ref = 1.4, 103.3
    ships = np.array(
        [
            [0.0, 0.0, 25.0],
            [24000.0, 4000.0, 25.0],
            [-18000.0, 17000.0, 25.0],
            [9000.0, -22000.0, 25.0],
        ]
    )
    dem = _RasterDEM(dem_path)
    bounds = _bounds(ships, args.min_alt, args.max_alt)

    print(f"Module: {Path(args.module).resolve()}")
    print(f"DEM: {dem_path.resolve()} (EPSG:4326, {dem.width}x{dem.height})")
    print(f"Fixed geometry: {len(ships)} ships; reference ({lat_ref}, {lon_ref})")
    print("Search bounds: " + ", ".join(
        f"[{low:.0f}, {high:.0f}]" for low, high in bounds
    ))
    for east in bounds[0]:
        for north in bounds[1]:
            lat, lon, _ = module.enu_to_lla(east, north, 0.0, lat_ref, lon_ref, 0.0)
            if not (
                dem.bounds.left <= lon <= dem.bounds.right
                and dem.bounds.bottom <= lat <= dem.bounds.top
            ):
                raise ValueError(
                    f"Search bounds extend outside DEM coverage: "
                    f"{dem.bounds}; reference point is ({lat_ref}, {lon_ref})"
                )

    dimensions = tuple(args.grid)
    grid_score, grid_position, grid_link_length_km, grid_seconds = _grid_search(
        module, ships, dem, lat_ref, lon_ref, bounds, dimensions
    )
    print(
        f"\nGrid {dimensions[0]}x{dimensions[1]}x{dimensions[2]} "
        f"({np.prod(dimensions)} points): score={grid_score:.1f}, "
        f"UAV link length={grid_link_length_km:.2f} km, "
        f"{_format_position(grid_position)}, {grid_seconds:.2f}s"
    )

    results_by_budget = {}
    for budget in args.budgets:
        results = []
        print(f"\nDE runs with time_budget_s={budget:g}")
        for seed in args.seeds:
            started = time.perf_counter()
            result = module.place_uavs(
                ships,
                dem,
                lat_ref,
                lon_ref,
                min_alt_m=args.min_alt,
                max_alt_m=args.max_alt,
                maxiter=args.maxiter,
                popsize=args.popsize,
                random_state=seed,
                time_budget_s=budget,
            )
            elapsed = time.perf_counter() - started
            position = np.asarray(result["uavs"][0]["uav_enu"], dtype=float)
            score, link_length_km = _score_position(
                module, ships, position, dem, lat_ref, lon_ref
            )
            in_bounds = _within_bounds(position, bounds)
            results.append(
                {
                    "seed": seed,
                    "score": score,
                    "position": position,
                    "elapsed": elapsed,
                    "timed_out": result["timed_out"],
                    "in_bounds": in_bounds,
                }
            )
            print(
                f"  seed={seed:>4} score={score:>10.1f} "
                f"grid_gap={score - grid_score:>10.1f} "
                f"UAV links={link_length_km:.2f} km "
                f"{_format_position(position)} "
                f"within_bounds={in_bounds} timed_out={result['timed_out']} "
                f"elapsed={elapsed:.2f}s"
            )
            if not in_bounds:
                raise RuntimeError(
                    f"optimizer returned an out-of-bounds placement for seed {seed}"
                )

        scores = np.array([item["score"] for item in results])
        positions = np.array([item["position"] for item in results])
        position_distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2
        )
        print(
            f"  score min/median/max="
            f"{scores.min():.1f}/{np.median(scores):.1f}/{scores.max():.1f}; "
            f"range={scores.max() - scores.min():.1f}; "
            f"max placement separation="
            f"{position_distances.max() / 1000:.2f} km"
        )
        results_by_budget[budget] = results

    if len(results_by_budget) > 1:
        print("\nBudget comparison (same seed, same geometry):")
        reference_budget = args.budgets[0]
        reference = {r["seed"]: r for r in results_by_budget[reference_budget]}
        for budget in args.budgets[1:]:
            comparison = {r["seed"]: r for r in results_by_budget[budget]}
            common_seeds = [seed for seed in args.seeds
                            if seed in reference and seed in comparison]
            print(f"  {reference_budget:g}s vs {budget:g}s:")
            for seed in common_seeds:
                short, long = reference[seed], comparison[seed]
                separation = np.linalg.norm(
                    short["position"] - long["position"]
                )
                print(
                    f"    seed={seed:>4} score delta="
                    f"{long['score'] - short['score']:>10.1f}; "
                    f"position delta={separation / 1000:.2f} km"
                )

    # A one-ship geometry has no placement preference; it checks that the
    # optimizer still returns a feasible candidate when horizontal span is zero.
    single_ship = np.array([[0.0, 0.0, 25.0]])
    degenerate_bounds = _bounds(single_ship, args.min_alt, args.max_alt)
    degenerate = module.place_uavs(
        single_ship,
        dem,
        lat_ref,
        lon_ref,
        min_alt_m=args.min_alt,
        max_alt_m=args.max_alt,
        maxiter=min(args.maxiter, 3),
        popsize=args.popsize,
        random_state=args.seeds[0],
        time_budget_s=max(args.budgets[0], 0.1),
    )
    degenerate_position = np.asarray(
        degenerate["uavs"][0]["uav_enu"], dtype=float
    )
    print(
        "\nDegenerate one-ship bounds check: "
        f"{_format_position(degenerate_position)}; "
        f"bounds={degenerate_bounds}; "
        f"within_bounds={_within_bounds(degenerate_position, degenerate_bounds)}"
    )
    if not _within_bounds(degenerate_position, degenerate_bounds):
        raise RuntimeError("optimizer returned an out-of-bounds degenerate placement")


if __name__ == "__main__":
    main()
