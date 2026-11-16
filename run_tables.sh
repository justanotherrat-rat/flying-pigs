#!/usr/bin/env bash
# Re-run everything behind Tables 2-5 from a clean, committed state.
#   NS3_ROOT=~/ns-3-dev N=50 SEED=1000 ./run_tables.sh
# Results go to results/<commit>/ ; nothing in sim-out/ or replications/ is touched or read.
set -euo pipefail
cd "$(dirname "$0")"
NS3_ROOT="${NS3_ROOT:-$HOME/ns-3-dev}"; N="${N:-50}"; SEED="${SEED:-1000}"
ARMS="${ARMS:-midpoint grid de1 de10 de60}"

if [ -n "$(git status --porcelain -- simulation testing test ':!*__pycache__*')" ]; then
  echo "Uncommitted changes in simulation/ testing/ test/ - commit first so the results map to a commit." >&2
  exit 1
fi
R="$PWD/results/$(git rev-parse --short HEAD)"
if [ -e "$R" ]; then echo "$R already exists - move it away; results are never mixed." >&2; exit 1; fi
mkdir -p "$R"
git rev-parse HEAD > "$R/commit.txt"
{ date -u; uname -a; python3 --version; grep -m1 'model name' /proc/cpuinfo || true; nproc; } > "$R/machine.txt" 2>&1
echo "N=$N SEED=$SEED ARMS=$ARMS" > "$R/params.txt"

echo "== 0. test gate (includes ITU-R P.527-6 verification) =="
python3 -m pytest test -m "not slow" -q

echo "== 1. Tables 2 and 4: proxy-score benchmark (idle machine, one process) =="
python3 testing/placement_benchmark.py --n "$N" --seed "$SEED" --out "$R/placement_benchmark.csv" \
  2>&1 | grep -v 'No DEM elevation' | tee "$R/placement_benchmark.log"

echo "== 2. Tables 3 and 5: ns-3 arms =="
TABLE_ARGS=()
for arm in $ARMS; do
  case "$arm" in
    midpoint) FLAGS=(--placement-mode midpoint) ;;
    grid)     FLAGS=(--placement-mode grid --grid 9 9 5) ;;
    de*)      FLAGS=(--placement-mode optimized --placement-budget-s "${arm#de}") ;;
    *) echo "unknown arm $arm" >&2; exit 1 ;;
  esac
  echo "-- arm: $arm --"
  (cd simulation && python3 run_replications.py --ns3-root "$NS3_ROOT" --n "$N" --seed "$SEED" \
      "${FLAGS[@]}" --out-prefix "${arm}_" --work-dir "$R/$arm") 2>&1 | grep -v 'No DEM elevation' | tee "$R/$arm.log"
  GOT=$(( $(wc -l < "$R/$arm/replication_results.csv") - 1 ))
  if [ "$GOT" -ne "$N" ]; then echo "ARM $arm: only $GOT/$N scenarios succeeded - check $R/$arm.log" >&2; exit 1; fi
  TABLE_ARGS+=(--arm "$arm=$R/$arm")
done

echo "== 3. tables =="
python3 testing/make_tables.py --bench "$R/placement_benchmark.csv" "${TABLE_ARGS[@]}" --out "$R/tables"
echo "Done. Report tables: $R/tables/tables.md"