#!/bin/bash
#SBATCH --job-name=JOB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mgallo@sissa.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu2
#SBATCH --output=./log_opt/%x.o%A-%a
#SBATCH --error=./log_opt/%x.o%A-%a

task=${1}

if [ "$task" -eq 1 ]; then
  echo "Running primitive NLP NTP task"
  num_samples=200000
  sequence_length=10
  context_window=3
  vocab_size=78
  embedding_dim=50
  embedding_model='glove.6B.50d'
  seed=42
  distr_param=1.1
  temperature=3

  python -u run_task.py \
    --task $task \
    --num_samples $num_samples \
    --sequence_length $sequence_length \
    --context_window $context_window \
    --vocab_size $vocab_size \
    --embedding_dim $embedding_dim \
    --embedding_model $embedding_model \
    --seed $seed \
    --distr_param $distr_param \
    --temperature $temperature

elif [ "$task" -eq 2 ]; then
  echo "Running primitive NLP summing task"
  
  num_samples=200000
  sequence_length=10
  context_window=3
  vocab_size=78
  embedding_dim=50
  embedding_model='glove.6B.50d'
  seed=42
  distr_param=1.1
  temperature=3

  python -u run_task.py \
    --task $task \
    --num_samples $num_samples \
    --sequence_length $sequence_length \
    --context_window $context_window \
    --vocab_size $vocab_size \
    --embedding_dim $embedding_dim \
    --embedding_model $embedding_model \
    --seed $seed \
    --distr_param $distr_param \
    --temperature $temperature

elif [ "$task" -eq 3 ]; then
  echo "Running Next Histogram task"
  
  num_samples=50000
  sequence_length=10
  vocab_size=15
  seed=42
  n_classes=7

  python -u run_task.py \
    --task $task \
    --num_samples $num_samples \
    --sequence_length $sequence_length \
    --vocab_size $vocab_size \
    --seed $seed \

fi