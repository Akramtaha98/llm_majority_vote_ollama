#!/usr/bin/env bash
cd "$(dirname "$0")/.."
echo "main: $(ls runs/unified/*_rep*.csv 2>/dev/null | wc -l)/18 | verbal: $(ls runs/unified/*_verbal.csv 2>/dev/null | wc -l)/6 | temp: $(ls runs/unified_T*/*.csv 2>/dev/null | wc -l)/18"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
tail -qn1 logs/*.log 2>/dev/null | cut -c1-140 | tail -6
