#!/bin/bash
#SBATCH --job-name=primitive_NLP_NTP_dataset_n_smpl50000__seq_len10__cont_win10__v_size78__emb_dim50__emb_typeglove.6B.50d__seed42__d_par2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mgallo@sissa.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu2
#SBATCH --output=./log_opt/%x.o%A-%a
#SBATCH --error=./log_opt/%x.o%A-%a

python -u run_task.py