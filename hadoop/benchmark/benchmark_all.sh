#!/bin/bash
# Runs all four Hadoop Streaming jobs back-to-back and records wall-clock
# time for each, then prints a summary table for the paper's results section.
#
# Run from the project root:
#   bash hadoop/benchmark/benchmark_all.sh
#
# Prerequisites:
#   - Hadoop loaded:  module load hadoop
#   - Data + models on HDFS:  bash hadoop/setup_hdfs.sh

set -e

RESULTS_FILE="hadoop/benchmark/hadoop_timing_results.txt"
echo "Model,WallClockSeconds" > "$RESULTS_FILE"

run_job() {
    local label="$1"
    local script="$2"
    echo ""
    echo "======================================================"
    echo " Running: $label"
    echo "======================================================"
    local start end elapsed
    start=$(date +%s)
    bash "$script"
    end=$(date +%s)
    elapsed=$((end - start))
    echo "$label,$elapsed" >> "$RESULTS_FILE"
    echo "  --> $label finished in ${elapsed}s"
}

run_job "Logistic_Regression" "hadoop/logistic_regression/run.sh"
run_job "Random_Forest"       "hadoop/random_forest/run.sh"
run_job "XGBoost"             "hadoop/xgboost/run.sh"
run_job "KNN"                 "hadoop/knn/run.sh"

echo ""
echo "======================================================"
echo " Hadoop timing summary"
echo "======================================================"
column -t -s',' "$RESULTS_FILE"
echo ""
echo "Compare against hadoop/benchmark/timing_results.txt"
echo "(serial baseline) to compute speedup ratios."
