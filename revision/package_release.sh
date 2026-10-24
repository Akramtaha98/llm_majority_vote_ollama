#!/usr/bin/env bash
# Build the per-call data release (attach the tarball to the GitHub release v2.0-revision) and copy the small result tables into results/.
# Run from the repo root on the machine that holds runs/.
set -euo pipefail; cd "$(dirname "$0")/.."
mkdir -p results release
[ -d runs/unified_T0.0 ] && [ ! -d runs/unified_T0.01 ] && mv runs/unified_T0.0 runs/unified_T0.01   # the near-greedy sweep is run at T=0.01
for t in unified_tables:tables_main tables_32b:tables_32b tables_70b:tables_70b; do src=runs/${t%%:*}; dst=results/${t##*:}; [ -d "$src" ] && rm -rf "$dst" && cp -r "$src" "$dst"; done
tar -czf release/unified_runs_v2.0-revision.tar.gz $(ls -d runs/unified runs/unified_T* runs/unified_big_* 2>/dev/null)
( cd release && sha256sum unified_runs_v2.0-revision.tar.gz > unified_runs_v2.0-revision.sha256 )
ls -la release results
