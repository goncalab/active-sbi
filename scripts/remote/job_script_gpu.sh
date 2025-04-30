#!/bin/bash
#SBATCH --gres=gpu:1          # one GPU
#SBATCH --mem=8G              # 8 GiB RAM
#SBATCH --time=0-01:00        # 1 hour (DD-HH:MM)
#SBATCH --job-name=active_sbi # (optional) job name
#SBATCH --output=%x-%j.out    # (optional) Slurm-captured stdout/stderr

# Activate conda environment
conda activate hayden

# ----- User variables -----
CONFIG="TWO_MOONS"
CODE_DIR="/data/groups/nerf/pedro.goncalves/hayden/active-sbi"
OUTPUT_DIR="$CODE_DIR/results/$CONFIG"
SCRIPT_FILE="$CODE_DIR/scripts/run_experiments.py"
CONFIG_FILE="$CODE_DIR/configs/$CONFIG.yaml"
# --------------------------

mkdir -p "$OUTPUT_DIR"

# Run the experiment, streaming stdout to both Slurm and your own log
python "$SCRIPT_FILE" "$CONFIG_FILE" > "$OUTPUT_DIR/results.log"
