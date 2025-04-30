#!/bin/bash

sbatch --job-name=active_sbi --partition=gp_64C_128T_512GB --account=c09_pi_pedro_goncalves --time=250 job_script_cpu.sh
