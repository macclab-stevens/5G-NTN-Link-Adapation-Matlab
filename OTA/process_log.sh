#!/bin/bash

# Script to process log files and extract values
# Usage: ./process_log.sh <log_file_path>

LOG_FILE="$1"

if [ -z "$LOG_FILE" ]; then
    echo "Usage: $0 <log_file_path>"
    echo "Example: $0 ./bler01/gnb-bler01.log"
    exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file '$LOG_FILE' not found!"
    exit 1
fi

echo "Processing log file: $LOG_FILE"

# Extract directory and filename
DIR=$(dirname "$LOG_FILE")
BASENAME=$(basename "$LOG_FILE" .log)

echo "Directory: $DIR"
echo "Basename: $BASENAME"

# You can add more processing here based on what value you want to extract
# For example, count lines, extract timestamps, etc.

echo "Log file contains $(wc -l < "$LOG_FILE") lines"

# If you want to run the Python processor on this file:
PYTHON_SCRIPT="../process_sched_mac_logs_optimized.py"
if [ -f "$PYTHON_SCRIPT" ]; then
    echo "Running Python processor..."
    python3 "$PYTHON_SCRIPT" --log-file "$LOG_FILE" --output "${DIR}/${BASENAME}-processed.csv" --verbose
else
    echo "Python script not found at $PYTHON_SCRIPT"
fi
