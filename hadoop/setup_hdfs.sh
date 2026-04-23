#!/bin/bash
# One-time HDFS setup: create directories and upload cleaned_data.csv.
# Run this after SSHing into CWRU HPCC, before run_inference.sh.
# Usage: bash hadoop/setup_hdfs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

HDFS_BASE=/user/mxf504/diabetes
LOCAL_DATA="${PROJECT_DIR}/cleaned_data.csv"

if [[ ! -f "$LOCAL_DATA" ]]; then
    echo "ERROR: cleaned_data.csv not found at ${LOCAL_DATA}"
    exit 1
fi

echo "Creating HDFS directory: ${HDFS_BASE}"
hdfs dfs -mkdir -p "$HDFS_BASE"

echo "Uploading cleaned_data.csv to HDFS..."
hdfs dfs -put -f "$LOCAL_DATA" "${HDFS_BASE}/cleaned_data.csv"

echo ""
echo "HDFS contents:"
hdfs dfs -ls "$HDFS_BASE"
echo ""
echo "Setup complete. Run: bash hadoop/run_inference.sh"
