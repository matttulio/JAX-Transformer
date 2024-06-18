# ---
# # Training a single Layer of Attention on the Histogram Task may lead to two solutions
# 
# This notebook shows how to train and evaluate single-layer transformers with dot-product attention and trained positional encodings.
# 
# - Part 1: Define data and model architecture
# - Part 2: Training of models with different settings on the attention layer (positional, semantic, or both)
# - Part 3: Introspecting the attention layers for some input sequences
# - Part 4: Checking whether the parameter values with the frozen weights stay close to their original position in the unfrozen weight space

from jax import random
from torch.utils.data import random_split
import pandas as pd
from src.transformer import *
from src.benchmarks import *
import os
import cloudpickle
import pickle
import argparse

# Determine environment (e.g., SSH or local)
running_on_cluster = 'SLURM_JOB_ID' in os.environ

if running_on_cluster:

    print("Running on remote cluster")

    parser = argparse.ArgumentParser(description='Run task with various parameters.')
    parser.add_argument('--task', type=int, required=True, help='Task number')
    args, remaining_args = parser.parse_known_args()
    task = args.task
else:

    print("Running locally")

    #task = 1  # Load primitive NLP NTP dataset
    #task = 2  # Load primitive NLP dataset
    task = 3  # Load NextHistogramTask dataset


data_path = 'Datasets/Data'

if(task == 1):

    print("PRIMITIVE NLP NTP TASK \n")

    if running_on_cluster:
        parser.add_argument('--num_samples', type=int, required=True, help='Number of samples')
        parser.add_argument('--sequence_length', type=int, required=True, help='Sequence length')
        parser.add_argument('--context_window', type=int, required=True, help='Context window')
        parser.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size')
        parser.add_argument('--embedding_dim', type=int, required=True, help='Embedding dimension')
        parser.add_argument('--embedding_model', type=str, required=True, help='Embedding model')
        parser.add_argument('--seed', type=int, required=True, help='Seed value')
        parser.add_argument('--distr_param', type=str, required=True, help='Distribution parameter')
        args = parser.parse_args()

        num_samples = args.num_samples
        sequence_length = args.sequence_length
        context_window = args.context_window
        vocab_size = args.vocab_size
        embedding_dim = args.embedding_dim
        embedding_path = 'Datasets/glove/glove.6B.50d.txt'
        embedding_model = args.embedding_model
        seed = args.seed

        try:
            distr_param = int(args.distr_param)
        except ValueError:
            distr_param = float(args.distr_param)

        n_classes = vocab_size + 1

    else:

        num_samples = 50000
        sequence_length = 10
        context_window = 10
        vocab_size = round(sequence_length * 7.8125)
        vocab = list(range(vocab_size))
        embedding_dim = 50
        embedding_path = 'Datasets/glove/glove.6B.50d.txt'
        embedding_model = 'glove.6B.50d'
        seed = 42
        distr_param = 2
        n_classes = vocab_size + 1

    

    file_name = f"primitive_NLP_NTP_dataset_n_smpl{num_samples}__seq_len{sequence_length}__cont_win{context_window}__" \
    + f"v_size{vocab_size}__emb_dim{embedding_dim}__emb_type{embedding_model}__seed{seed}__d_par{distr_param}"
    file_ext = '.pkl'
    save_dir = os.path.join('Empirics', file_name)
    

elif(task == 2):

    print("PRIMITIVE NLP SUMMING TASK \n")

    if running_on_cluster:
        parser.add_argument('--num_samples', type=int, required=True, help='Number of samples')
        parser.add_argument('--sequence_length', type=int, required=True, help='Sequence length')
        parser.add_argument('--context_window', type=int, required=True, help='Context window')
        parser.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size')
        parser.add_argument('--embedding_dim', type=int, required=True, help='Embedding dimension')
        parser.add_argument('--embedding_model', type=str, required=True, help='Embedding model')
        parser.add_argument('--seed', type=int, required=True, help='Seed value')
        parser.add_argument('--distr_param', type=str, required=True, help='Distribution parameter')
        args = parser.parse_args()

        num_samples = args.num_samples
        sequence_length = args.sequence_length
        context_window = args.context_window
        vocab_size = args.vocab_size
        embedding_dim = args.embedding_dim
        embedding_path = 'Datasets/glove/glove.6B.50d.txt'
        embedding_model = args.embedding_model
        seed = args.seed

        try:
            distr_param = int(args.distr_param)
        except ValueError:
            distr_param = float(args.distr_param)
            
        n_classes = 2

    else:

        num_samples = 50000
        sequence_length = 10
        context_window = 10
        vocab_size = round(sequence_length * 7.8125)
        vocab = list(range(vocab_size))
        embedding_dim = 50
        embedding_path = 'Datasets/glove/glove.6B.50d.txt'
        embedding_model = 'glove.6B.50d'
        seed = 42
        distr_param = 2
        n_classes = 2


    file_name = f"primitive_NLP_dataset_n_smpl{num_samples}__seq_len{sequence_length}__cont_win{context_window}__" \
    + f"v_size{vocab_size}__emb_dim{embedding_dim}__emb_type{embedding_model}__seed{seed}__d_par{distr_param}"
    file_ext = '.pkl'
    save_dir = os.path.join('Empirics', file_name)


elif(task == 3):

    print("NEXT HISTOGRAM TASK \n")

    if running_on_cluster:
        parser.add_argument('--num_samples', type=int, required=True, help='Number of samples')
        parser.add_argument('--sequence_length', type=int, required=True, help='Sequence length')
        parser.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size')
        parser.add_argument('--seed', type=int, required=True, help='Seed value')
        args = parser.parse_args()

        num_samples = args.num_samples
        sequence_length = args.sequence_length
        vocab_size = args.vocab_size
        seed = args.seed
        n_classes = 7
    
    else:
        
        num_samples = 50000
        sequence_length = 10
        vocab_size = 15
        seed = 42
        n_classes = 7


    file_name = f'NextHistogramDataset_n_smpl{num_samples}__seq_len{sequence_length}__v_size{vocab_size}__seed{seed}'
    file_ext = '.pkl'
    save_dir = os.path.join('Empirics', file_name)


print(f"{file_name} \n")
file_name = file_name + file_ext

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(os.path.join(data_path, file_name), "rb") as f: 
    dataset = pickle.load(f)

# Define the length, maximum value, and number of samples
n_classes = dataset.n_classes
vocab_size = dataset.vocab_size
seq_len = dataset.seq_len
num_samples = dataset.num_samples

# Split the dataset into train, test, and validation sets
seed = 42
generator = torch.Generator().manual_seed(seed)


train_ratio, val_ratio = 0.7, 0.3

train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, generator=generator)

# Convert dataset to JAX arrays
train_dataset = [convert_batch_to_jax(batch) for batch in train_dataloader]
val_dataset = [convert_batch_to_jax(batch) for batch in val_dataloader]

hidden_dimension_fc = 128
model_dim = 64
batch_size = 32
n_runs = 5
#model_types = ['only_pos', 'only_sem']
model_types = ['only_sem', 'only_pos']
n_epochs = 200

results = []
dummy_input = np.ones(shape=(batch_size, seq_len), dtype=np.int8)
learning_rate = 1e-3

for model_type in model_types:
    print(f'Running {model_type}...')
    for i in range(n_runs):

        print(f"Run: {i}")

        rng = random.PRNGKey(i)

        transformer = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, seq_len, model_type)
        state = init_train_state(transformer, rng, dummy_input, learning_rate)
        
        with open(os.path.join(save_dir, f'run_{i}_initmodel_{model_type}_orig.pkl'), "wb") as file:
            cloudpickle.dump(state, file)

        train_minibatch_first_metrics, eval_minibatch_first_metrics = eval_init(train_dataset, val_dataset, state)

        first_train_loss = {'loss': train_minibatch_first_metrics['loss']}
        first_val_loss = {'loss': eval_minibatch_first_metrics['loss']}
        first_val_acc = {'accuracy': eval_minibatch_first_metrics['accuracy']}
        
        trained_state, train_minibatch_metrics, val_minibatch_metrics, train_epoch_metrics, val_epoch_metrics = train_and_evaluate(train_dataset, val_dataset, state, n_epochs)
        with open(os.path.join(save_dir, f'run_{i}_model_{model_type}_orig.pkl'), "wb") as file:
            cloudpickle.dump(trained_state, file)

        train_epoch_metrics.insert(0, first_train_loss)
        val_epoch_metrics.insert(0, first_val_loss)
        val_epoch_metrics.insert(0, first_val_acc)

        results.append({
          'model_type': model_type,
            'train_losses': [float(metrics['loss']) for metrics in train_epoch_metrics if 'loss' in metrics],
            'val_losses': [float(metrics['loss']) for metrics in val_epoch_metrics if 'loss' in metrics],
            'val_acc': [float(metrics['accuracy']) for metrics in val_epoch_metrics if 'accuracy' in metrics],
            'run':i,
        })

    print('Done')

pd.DataFrame(results).to_csv(os.path.join(save_dir, 'frozen_transformer_result.csv'), index=False)


n_epochs = n_epochs // 2
learning_rate = 1e-4
reparameterized_transformers = []

for r in range(n_runs):

    rng = random.PRNGKey(r)

    for model_type in model_types:
  
        print(r, model_type)

        with open(os.path.join(save_dir, f'run_{r}_model_{model_type}_orig.pkl'), "rb") as file:
            orig_state = cloudpickle.load(file)

        rep_trans_state = reparameterize(vocab_size, model_dim, hidden_dimension_fc, n_classes, seq_len, orig_state, model_type, dummy_input, learning_rate, rng)

        train_minibatch_first_metrics, eval_minibatch_first_metrics = eval_init(train_dataset, val_dataset, rep_trans_state)
        
        first_train_loss = {'loss': train_minibatch_first_metrics['loss']}
        first_val_loss = {'loss': eval_minibatch_first_metrics['loss']}
        first_val_acc = {'accuracy': eval_minibatch_first_metrics['accuracy']}

        with open(os.path.join(save_dir, f'run_{r}_model_{model_type}_repar.pkl'), "wb") as file:
            cloudpickle.dump(trained_state, file)

        trained_state, train_minibatch_metrics, val_minibatch_metrics, train_epoch_metrics, val_epoch_metrics = train_and_evaluate(train_dataset, val_dataset, rep_trans_state, n_epochs)

        with open(os.path.join(save_dir, f'run_{r}_model_{model_type}_retrained.pkl'), "wb") as file:
            cloudpickle.dump(trained_state, file)

        train_epoch_metrics.insert(0, first_train_loss)
        val_epoch_metrics.insert(0, first_val_loss)
        val_epoch_metrics.insert(0, first_val_acc)

        reparameterized_transformers.append({
          'model_type': model_type,
            'train_losses': [float(metrics['loss']) for metrics in train_epoch_metrics if 'loss' in metrics],
            'val_losses': [float(metrics['loss']) for metrics in val_epoch_metrics if 'loss' in metrics],
            'val_acc': [float(metrics['accuracy']) for metrics in val_epoch_metrics if 'accuracy' in metrics],
            'run':r,
        })

        print('Done')

pd.DataFrame(reparameterized_transformers).to_csv(os.path.join(save_dir, 'reparameterized_transformers.csv'), index=False)