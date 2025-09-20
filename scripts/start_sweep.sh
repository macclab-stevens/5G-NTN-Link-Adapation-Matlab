#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for the MATLAB NTN sweep
#
# Usage examples:
#   ./scripts/start_sweep.sh quick-serial
#   ./scripts/start_sweep.sh serial
#   ./scripts/start_sweep.sh threads 8
#   ./scripts/start_sweep.sh processes 16
#
# Env overrides:
#   MATLAB_BIN   (default: $HOME/mathworks/bin/matlab)
#   OUTPUT_DIR   (default: /home/snrGrp2024_01/5G-NTN-Link-Adapation-Matlab/ML/data/generated)
#   LOG_DIR      (default: /home/snrGrp2024_01/5G-NTN-Link-Adapation-Matlab/logs)
#   SNR_RANGE    (default: -12:3:21)
#   FRAMES       (default: 8000; quick uses 1000)
#   BLER_WIN     (default: 1000)
#   TBLER        (default: 0.01)
#   ALTITUDE     (default: 600e3)
#   SEEDS        (default: 1:4)

REPO_DIR="/home/snrGrp2024_01/5G-NTN-Link-Adapation-Matlab"
MATLAB_BIN=${MATLAB_BIN:-$HOME/mathworks/bin/matlab}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/ML/data/generated}
LOG_DIR=${LOG_DIR:-$REPO_DIR/logs}
SNR_RANGE=${SNR_RANGE:--12:3:21}
FRAMES=${FRAMES:-8000}
BLER_WIN=${BLER_WIN:-1000}
TBLER=${TBLER:-0.01}
ALTITUDE=${ALTITUDE:-600e3}
SEEDS=${SEEDS:-1:4}

mkdir -p "$LOG_DIR"

# Helper: kill lingering MATLAB processes cleanly
kill_matlab() {
  echo "Killing MATLAB processes..."
  pkill -TERM -f 'MATLABWorker' || true
  pkill -TERM -f '[m]atlab' || true
  sleep 2
  pkill -KILL -f 'MATLABWorker' || true
  pkill -KILL -f '[m]atlab' || true
}

case "${1:-quick-serial}" in
  kill)
    kill_matlab
    ;;
  quick-serial)
    # Minimal smoke: serial, no pool, tiny run
    nohup "$MATLAB_BIN" -batch \
      "addpath('$REPO_DIR/Matlab'); run_link_adaptation_sweeps('UseParallel',false,'SNRRange',-6,'Frames',1000,'Seeds',[1:2],'BlerWindow',$BLER_WIN,'TargetBler',$TBLER,'Altitude',$ALTITUDE,'LogEveryNSlots',200,'OutputDir','$OUTPUT_DIR')" \
      > "$LOG_DIR/run_$(date +%F_%H%M%S).log" 2>&1 &
    ;;
  serial)
    nohup "$MATLAB_BIN" -batch \
      "addpath('$REPO_DIR/Matlab'); run_link_adaptation_sweeps('UseParallel',false,'SNRRange',[$SNR_RANGE],'Frames',$FRAMES,'Seeds',[$SEEDS],'BlerWindow',$BLER_WIN,'TargetBler',$TBLER,'Altitude',$ALTITUDE,'LogEveryNSlots',500,'OutputDir','$OUTPUT_DIR')" \
      > "$LOG_DIR/run_$(date +%F_%H%M%S).log" 2>&1 &
    ;;
  threads)
    workers=${2:-}
    nohup "$MATLAB_BIN" -batch \
      "addpath('$REPO_DIR/Matlab'); run_link_adaptation_sweeps('UseParallel',true,'PoolType','Threads','SNRRange',[$SNR_RANGE],'Frames',$FRAMES,'Seeds',[$SEEDS],'BlerWindow',$BLER_WIN,'TargetBler',$TBLER,'Altitude',$ALTITUDE,'LogEveryNSlots',500,'OutputDir','$OUTPUT_DIR')" \
      > "$LOG_DIR/run_$(date +%F_%H%M%S).log" 2>&1 &
    ;;
  processes)
    workers=${2:-8}
    # Allow MATLAB/BLAS to choose threading (tune via OMP/MKL env vars if desired)
    nohup "$MATLAB_BIN" -batch \
      "addpath('$REPO_DIR/Matlab'); run_link_adaptation_sweeps('UseParallel',true,'PoolType','Processes','Workers',$workers,'SNRRange',[$SNR_RANGE],'Frames',$FRAMES,'Seeds',[$SEEDS],'BlerWindow',$BLER_WIN,'TargetBler',$TBLER,'Altitude',$ALTITUDE,'LogEveryNSlots',100,'OutputDir','$OUTPUT_DIR')" \
      > "$LOG_DIR/run_$(date +%F_%H%M%S).log" 2>&1 &
    ;;
  serial-fanout)
    # Launch multiple independent serial jobs with disjoint seed ranges
    # Usage: serial-fanout JOBS SEED_START SEED_END
    jobs=${2:-}
    seed_start=${3:-}
    seed_end=${4:-}
    if [[ -z "$jobs" || -z "$seed_start" || -z "$seed_end" ]]; then
      echo "Usage: $0 serial-fanout JOBS SEED_START SEED_END" >&2
      exit 2
    fi
    if ! [[ "$jobs" =~ ^[0-9]+$ && "$seed_start" =~ ^[0-9]+$ && "$seed_end" =~ ^[0-9]+$ ]]; then
      echo "JOBS/SEED_START/SEED_END must be integers" >&2
      exit 2
    fi
    total=$(( seed_end - seed_start + 1 ))
    if (( total <= 0 )); then
      echo "Invalid seed range: start=$seed_start end=$seed_end" >&2
      exit 2
    fi
    chunk=$(( (total + jobs - 1) / jobs ))
    echo "Launching $jobs serial jobs across seeds [$seed_start:$seed_end] (chunk=$chunk)"
    for ((i=0; i<jobs; i++)); do
      s=$(( seed_start + i*chunk ))
      e=$(( s + chunk - 1 ))
      if (( s > seed_end )); then break; fi
      if (( e > seed_end )); then e=$seed_end; fi
      SEED_RANGE="$s:$e"
      job_out="$OUTPUT_DIR/fanout_job$((i+1))"
      mkdir -p "$job_out"
      log="$LOG_DIR/run_fanout$((i+1))_$(date +%F_%H%M%S).log"
      echo "Job $((i+1)): Seeds [$SEED_RANGE] -> $job_out (log: $log)"
      nohup "$MATLAB_BIN" -batch \
        "addpath('$REPO_DIR/Matlab'); run_link_adaptation_sweeps('UseParallel',false,'SNRRange',[$SNR_RANGE],'Frames',$FRAMES,'BlerWindow',$BLER_WIN,'TargetBler',$TBLER,'Altitude',$ALTITUDE,'Seeds',[$SEED_RANGE],'LogEveryNSlots',500,'OutputDir','$job_out')" \
        > "$log" 2>&1 &
      sleep 0.2
    done
    ;;
  serial-fanout-windows)
    jobs=${2:-}
    if [[ -z "$jobs" ]]; then
      echo "Usage: $0 serial-fanout-windows JOBS" >&2
      exit 2
    fi
    if ! [[ "$jobs" =~ ^[0-9]+$ ]]; then
      echo "JOBS must be an integer" >&2
      exit 2
    fi
    win_list_str=$(echo "$BLER_WIN" | tr -d '[],' )
    read -r -a windows <<< "$win_list_str"
    total=${#windows[@]}
    if (( total == 0 )); then
      echo "BLER_WIN is empty; set BLER_WIN env" >&2
      exit 2
    fi
    chunk=$(( (total + jobs - 1) / jobs ))
    echo "Launching $jobs serial jobs across $total BLER windows (chunk=$chunk) with Seeds=[$SEEDS]"
    for ((i=0; i<jobs; i++)); do
      start=$(( i * chunk ))
      if (( start >= total )); then break; fi
      end=$(( start + chunk ))
      if (( end > total )); then end=$total; fi
      subset=("${windows[@]:start:end-start}")
      subset_str=$(printf "%s " "${subset[@]}")
      subset_str="[${subset_str% }]"
      job_out="$OUTPUT_DIR/fanout_win$((i+1))"
      mkdir -p "$job_out"
      log="$LOG_DIR/run_fanout_win$((i+1))_$(date +%F_%H%M%S).log"
      echo "Job $((i+1)): BLER windows $subset_str -> $job_out (log: $log)"
      nohup "$MATLAB_BIN" -batch \
        "addpath('$REPO_DIR/Matlab'); run_link_adaptation_sweeps('UseParallel',false,'SNRRange',[$SNR_RANGE],'Frames',$FRAMES,'Seeds',[$SEEDS],'BlerWindow',$subset_str,'TargetBler',$TBLER,'Altitude',$ALTITUDE,'LogEveryNSlots',500,'OutputDir','$job_out')" \
        > "$log" 2>&1 &
      sleep 0.2
    done
    ;;
  *)
    echo "Unknown mode: ${1:-}" >&2
    echo "Usage: $0 [kill|quick-serial|serial|threads|processes [N]|serial-fanout JOBS SEED_START SEED_END|serial-fanout-windows JOBS]" >&2
    exit 2
    ;;
esac

echo "Launched. Tail logs with: tail -f $LOG_DIR/$(ls -t $LOG_DIR | head -n1)"
