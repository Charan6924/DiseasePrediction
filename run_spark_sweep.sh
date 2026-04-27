#!/bin/bash

#SBATCH --account=csds312

#SBATCH --partition=markov_cpu

#SBATCH --job-name=diabetes-spark-sweep

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=8gb

#SBATCH --time=01:00:00

#SBATCH --output=logs/sweep-%j.out

#SBATCH --error=logs/sweep-%j.err



set -u

module load Spark/3.2.1-foss-2021b

source venv/bin/activate

mkdir -p logs results



RESULTS=results/spark_sweep_$(date +%Y%m%d_%H%M%S).txt

MODELS=("Logistic_Regression" "Random_Forest" "XGBoost")

CORES=(1 2 4)



echo "=== Spark sweep ===" | tee "$RESULTS"

echo "Started: $(date)"    | tee -a "$RESULTS"

echo "Node:    $(hostname)" | tee -a "$RESULTS"



for model in "${MODELS[@]}"; do

    pkl="models/${model}.pkl"

    for n in "${CORES[@]}"; do

        echo "--- Spark local[$n]: $model ---" | tee -a "$RESULTS"

        log="logs/spark_${model}_n${n}.log"

        spark-submit --master "local[$n]" --driver-memory 4G --files "$pkl" spark_inference.py "$model" > "$log" 2>&1

        grep -E "Wall-clock time|Accuracy|Recall|F1 Score" "$log" | tee -a "$RESULTS"

        echo | tee -a "$RESULTS"

    done

done



echo "=== Sweep complete: $(date) ===" | tee -a "$RESULTS"

