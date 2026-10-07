#!/usr/bin/env bash
cd "$(dirname "$0")/.."
echo "--- running jobs ---"; pgrep -af "run_information_experiments|eval_dataset|verbalized_confidence" | cut -c1-140
echo "--- finished files ---"
echo "verbal:      $(ls runs/verbalized/*_summary.csv 2>/dev/null | wc -l)/6 summaries"
echo "temperature: $(ls runs/temperature_extended/*.csv 2>/dev/null | wc -l)/24 CSVs"
echo "--- last log lines ---"; tail -n 2 logs/verbal_run.log logs/temp_run_*.log 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv 2>/dev/null
