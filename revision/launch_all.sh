#!/usr/bin/env bash
# from repo root: bash revision/launch_all.sh   -> 4 tmux sessions, one per GPU
cd "$(dirname "$0")/.."; mkdir -p logs
for g in 0 1 2 3; do
  tmux new-session -d -s mmv_gpu$g "bash revision/worker.sh $g revision/jobs.txt; bash revision/worker.sh $g revision/jobs_extra.txt 2>&1 | tee logs/worker$g.txt"
done
tmux ls
