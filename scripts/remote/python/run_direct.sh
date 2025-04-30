#!/bin/bash

CONFIG="TWO_MOONS"
CODE_DIR="/data/groups/nerf/pedro.goncalves/hayden/active-sbi"
CONFIG_DIR="$CODE_DIR/configs/$CONFIG.yaml"
OUTPUT_DIR="$CODE_DIR/results/$CONFIG"

mkdir -p "$OUTPUT_DIR"

python run_experiments.py "$CONFIG_DIR" "$OUTPUT_DIR" > "$OUTPUT_DIR/results.log"

