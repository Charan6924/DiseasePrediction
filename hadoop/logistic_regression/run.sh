#!/bin/bash
# Hadoop Streaming job — Logistic Regression inference
#
# Run from the project root on the HPC cluster:
#   bash hadoop/logistic_regression/run.sh
#
# Prerequisites:
#   - Hadoop loaded:  module load hadoop  (or equivalent)
#   - Data on HDFS:   bash hadoop/setup_hdfs.sh

set -e

MODEL_NAME="Logistic_Regression"
HDFS_BASE="/user/$USER/diabetes"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${HDFS_BASE}/output/${MODEL_NAME}_${TIMESTAMP}"

# Locate the Hadoop Streaming jar automatically
STREAMING_JAR=$(find "$HADOOP_HOME" -name "hadoop-streaming*.jar" 2>/dev/null | head -1)
if [ -z "$STREAMING_JAR" ]; then
    echo "ERROR: Could not find hadoop-streaming jar under HADOOP_HOME=$HADOOP_HOME"
    exit 1
fi

echo "=== Logistic Regression — Hadoop MapReduce ==="
echo "Streaming jar : $STREAMING_JAR"
echo "Output dir    : $OUTPUT_DIR"
echo ""

START=$(date +%s)

hadoop jar "$STREAMING_JAR" \
    -files hadoop/logistic_regression/mapper.py,hadoop/reducer.py,models/${MODEL_NAME}.pkl \
    -mapper  "python3 mapper.py" \
    -reducer "python3 reducer.py" \
    -numReduceTasks 1 \
    -input  "${HDFS_BASE}/data/brfss_diabetes_clean.csv" \
    -output "${OUTPUT_DIR}"

END=$(date +%s)
ELAPSED=$((END - START))

echo ""
echo "=== Results ==="
hadoop fs -cat "${OUTPUT_DIR}/part-00000"
echo ""
echo "Wall-clock time: ${ELAPSED}s"
echo "Output saved to HDFS: ${OUTPUT_DIR}"
