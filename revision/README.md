# Unified revision runs (Information, major revision)
Run from the repository root: `bash revision/setup.sh`, smoke test, then `bash revision/launch_all.sh`, `watch bash revision/monitor.sh`,
and finally `python revision/analyze_unified.py --runs runs/unified --out runs/unified_tables --boot 2000`.
