#!/bin/bash
#SBATCH --partition=gp_64C_128T_512GB
#SBATCH --mem=8G
#SBATCH --time=0-1:00

# Activate conda environment
conda activate hayden

# Define variables
CONFIG="TWO_MOONS"
CODE_DIR="/data/groups/nerf/pedro.goncalves/hayden/active-sbi"
OUTPUT_DIR="$CODE_DIR/results/$CONFIG"
SCRIPT_FILE="$CODE_DIR/scripts/run_experiments.py"
CONFIG_FILE="$CODE_DIR/configs/$CONFIG.yaml"

mkdir -p "$OUTPUT_DIR"

# Commands to execute using variables
python "$SCRIPT_FILE" "$CONFIG_FILE" > "$OUTPUT_DIR/results.log"
