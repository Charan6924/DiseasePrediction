#!/bin/bash
# Run this once from the HPC cluster to upload data and models to HDFS.
# Assumes: Hadoop is loaded (module load hadoop), and you're in the project root.
#
# Usage: bash hadoop/setup_hdfs.sh

set -e

HDFS_BASE="/user/$USER/diabetes"
DATA_LOCAL="brfss_diabetes_clean.csv/brfss_diabetes_clean.csv"
MODELS_LOCAL="models"

echo "=== Creating HDFS directories ==="
hadoop fs -mkdir -p ${HDFS_BASE}/data
hadoop fs -mkdir -p ${HDFS_BASE}/models
hadoop fs -mkdir -p ${HDFS_BASE}/output

echo "=== Uploading dataset ==="
hadoop fs -put -f ${DATA_LOCAL} ${HDFS_BASE}/data/brfss_diabetes_clean.csv

echo "=== Uploading model pickle files ==="
for pkl in ${MODELS_LOCAL}/*.pkl; do
    echo "  Uploading $pkl ..."
    hadoop fs -put -f "$pkl" ${HDFS_BASE}/models/
done

echo ""
echo "=== HDFS layout ==="
hadoop fs -ls -R ${HDFS_BASE}
echo ""
echo "Setup complete. HDFS base: ${HDFS_BASE}"
