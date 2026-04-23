#!/bin/bash
# Run Hadoop Streaming inference for all 4 trained models on CWRU HPCC.
# SSH into the cluster first, then: bash hadoop/run_inference.sh
#
# Prerequisites:
#   1. bash hadoop/install_deps.sh    (once per session)
#   2. bash hadoop/setup_hdfs.sh      (once, or after data changes)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Config ──────────────────────────────────────────────────────────────────
HDFS_BASE=/user/mxf504/diabetes
HDFS_INPUT="${HDFS_BASE}/cleaned_data.csv"
MODELS_DIR="${PROJECT_DIR}/models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PROJECT_DIR}/results/${TIMESTAMP}"

MODELS=(
    "Random_Forest"
    "Logistic_Regression"
    "XGBoost"
    "K-Nearest_Neighbors"
)

# ── Locate streaming jar ─────────────────────────────────────────────────────
STREAMING_JAR=$(find "${HADOOP_HOME}" -name "hadoop-streaming-*.jar" 2>/dev/null | head -1)
if [[ -z "$STREAMING_JAR" ]]; then
    echo "ERROR: hadoop-streaming jar not found under HADOOP_HOME=${HADOOP_HOME}"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "Streaming jar : $STREAMING_JAR"
echo "HDFS input    : $HDFS_INPUT"
echo "Local results : $RESULTS_DIR"
echo ""

# ── Verify input data exists in HDFS ────────────────────────────────────────
if ! hdfs dfs -test -e "$HDFS_INPUT"; then
    echo "ERROR: $HDFS_INPUT not found in HDFS. Run bash hadoop/setup_hdfs.sh first."
    exit 1
fi

# ── Run one Hadoop Streaming job per model ───────────────────────────────────
for MODEL in "${MODELS[@]}"; do
    MODEL_PKL="${MODELS_DIR}/${MODEL}.pkl"
    HDFS_OUTPUT="${HDFS_BASE}/predictions_${MODEL}_${TIMESTAMP}"

    if [[ ! -f "$MODEL_PKL" ]]; then
        echo "WARN: ${MODEL}.pkl not found at ${MODEL_PKL} — skipping."
        continue
    fi

    echo "========================================"
    echo "Model: ${MODEL}"
    echo "========================================"

    hdfs dfs -rm -r -f "$HDFS_OUTPUT"

    # The #model.pkl alias renames the distributed file on each worker node
    # so mapper.py can always open "model.pkl" regardless of which model runs.
    hadoop jar "$STREAMING_JAR" \
        -files "${SCRIPT_DIR}/mapper.py,${SCRIPT_DIR}/reducer.py,${MODEL_PKL}#model.pkl" \
        -input  "$HDFS_INPUT" \
        -output "$HDFS_OUTPUT" \
        -mapper  "python3 mapper.py" \
        -reducer "python3 reducer.py" \
        -numReduceTasks 1

    echo ""
    echo "--- ${MODEL} metrics ---"
    hdfs dfs -cat "${HDFS_OUTPUT}/part-00000" | tee "${RESULTS_DIR}/${MODEL}.txt"
    echo ""
done

echo "========================================"
echo "All jobs complete."
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Compare all models:"
echo "  python3 ${SCRIPT_DIR}/compare_results.py ${RESULTS_DIR}"
