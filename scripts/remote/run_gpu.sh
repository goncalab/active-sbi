#!/bin/bash

sbatch --job-name=active_sbi --partition=gpu_l40s_64C_128T_1TB --account=c09_pi_pedro_goncalves --gres=gpu:1 --time=250 job_script_gpu.sh
