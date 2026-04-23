#!/bin/bash
#SBATCH --job-name=ml_train
#SBATCH --partition=cgpudlw
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=/home/cxv166/DiseasePrediction/logs/%j_train.out
#SBATCH --error=/home/cxv166/DiseasePrediction/logs/%j_train.err

mkdir -p /home/cxv166/DiseasePrediction/logs/

uv run main.py
