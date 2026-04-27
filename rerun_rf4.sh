#!/bin/bash
#SBATCH --account=csds312
#SBATCH --partition=markov_cpu
#SBATCH --job-name=rf4-rerun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:15:00
#SBATCH --output=logs/rf4-%j.out
#SBATCH --error=logs/rf4-%j.err

set -u
module load Spark/3.2.1-foss-2021b
source venv/bin/activate

log="logs/spark_Random_Forest_n4.log"
spark-submit --master "local[4]" --driver-memory 8G --files models/Random_Forest.pkl spark_inference.py Random_Forest > "$log" 2>&1
grep -E "Wall-clock time|Accuracy|Recall|F1 Score" "$log"
