#!/bin/bash
#SBATCH --job-name=PNLPNTP_cw3_dp2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mgallo@sissa.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu2
#SBATCH --output=./log_opt/%x.o%A-%a
#SBATCH --error=./log_opt/%x.o%A-%a

python -u run_task.py