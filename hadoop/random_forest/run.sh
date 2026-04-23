#!/bin/bash
# Hadoop Streaming job — Random Forest inference
#
# Run from the project root on the HPC cluster:
#   bash hadoop/random_forest/run.sh
#
# NOTE: Random_Forest.pkl is ~1.9 GB.  Each mapper task loads its own copy
# into memory, so keep the mapper count low enough that the node's RAM isn't
# exhausted.  The -D flags below cap the job at 4 mappers and raise the
# per-mapper memory ceiling to 8 GB — adjust for your cluster.
#
# Prerequisites:
#   - Hadoop loaded:  module load hadoop  (or equivalent)
#   - Data on HDFS:   bash hadoop/setup_hdfs.sh

set -e

MODEL_NAME="Random_Forest"
HDFS_BASE="/user/$USER/diabetes"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${HDFS_BASE}/output/${MODEL_NAME}_${TIMESTAMP}"

STREAMING_JAR=$(find "$HADOOP_HOME" -name "hadoop-streaming*.jar" 2>/dev/null | head -1)
if [ -z "$STREAMING_JAR" ]; then
    echo "ERROR: Could not find hadoop-streaming jar under HADOOP_HOME=$HADOOP_HOME"
    exit 1
fi

echo "=== Random Forest — Hadoop MapReduce ==="
echo "Streaming jar : $STREAMING_JAR"
echo "Output dir    : $OUTPUT_DIR"
echo ""

START=$(date +%s)

hadoop jar "$STREAMING_JAR" \
    -D mapreduce.job.maps=4 \
    -D mapreduce.map.memory.mb=8192 \
    -D mapreduce.map.java.opts="-Xmx7168m" \
    -files hadoop/random_forest/mapper.py,hadoop/reducer.py,models/${MODEL_NAME}.pkl \
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
