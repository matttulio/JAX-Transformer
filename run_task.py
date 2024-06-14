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

task = 1  # Load primitive NLP dataset
#task = 2  # Load NextHistogramTask dataset
#task = 3  # Load primitive NLP NTP dataset

data_path = 'Datasets/Data'

if(task == 1):
    print("PRIMITIVE NLP TASK \n")
    file_name = 'primitive_NLP_dataset_n_smpl50000__seq_len10__cont_win3__'\
        'v_size78__emb_dim50__emb_typeglove.6B.50d__seed42__d_par2'
    file_ext = '.pkl'
    save_dir = os.path.join('Empirics', file_name)
elif(task == 2):
    print("NEXT HISTOGRAM TASK \n")
    file_name = 'NextHistogramDataset_n_smpl50000__seq_len10__v_size15__seed42'
    file_ext = '.pkl'
    save_dir = os.path.join('Empirics', file_name)
elif(task == 3):
    print("PPRIMITIVE NLP NTP TASK \n")
    file_name = 'primitive_NLP_NTP_dataset_n_smpl50000__seq_len10__cont_win10__'\
    'v_size78__emb_dim50__emb_typeglove.6B.50d__seed42__d_par2'
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
        
        trained_state, train_minibatch_metrics, val_minibatch_metrics, train_epoch_metrics, val_epoch_metrics = train_and_evaluate(train_dataset, val_dataset, state, n_epochs)
        with open(os.path.join(save_dir, f'run_{i}_model_{model_type}_orig.pkl'), "wb") as file:
            cloudpickle.dump(trained_state, file)

        #print(train_epoch_metrics)
        first_train_loss = {'loss': train_minibatch_metrics[0]['loss']}
        first_val_loss = {'loss': train_minibatch_metrics[0]['loss']}
        first_val_acc = {'accuracy': train_minibatch_metrics[0]['accuracy']}

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

        rep_trans_state = reparameterize(vocab_size, model_dim, hidden_dimension_fc, n_classes, seq_len, orig_state, model_type, rng)

        with open(os.path.join(save_dir, f'run_{r}_model_{model_type}_repar.pkl'), "wb") as file:
            cloudpickle.dump(trained_state, file)

        trained_state, train_minibatch_metrics, val_minibatch_metrics, train_epoch_metrics, val_epoch_metrics = train_and_evaluate(train_dataset, val_dataset, rep_trans_state, n_epochs)

        with open(os.path.join(save_dir, f'run_{r}_model_{model_type}_retrained.pkl'), "wb") as file:
            cloudpickle.dump(trained_state, file)

        first_train_loss = {'loss': train_minibatch_metrics[0]['loss']}
        first_val_loss = {'loss': train_minibatch_metrics[0]['loss']}
        first_val_acc = {'accuracy': train_minibatch_metrics[0]['accuracy']}

        train_epoch_metrics.insert(0, first_train_loss)
        val_epoch_metrics.insert(0, first_val_loss)
        val_epoch_metrics.insert(0, first_val_acc)

        results.append({
          'model_type': model_type,
            'train_losses': [float(metrics['loss']) for metrics in train_epoch_metrics if 'loss' in metrics],
            'val_losses': [float(metrics['loss']) for metrics in val_epoch_metrics if 'loss' in metrics],
            'val_acc': [float(metrics['accuracy']) for metrics in val_epoch_metrics if 'accuracy' in metrics],
            'run':r,
        })

        print('Done')

pd.DataFrame(reparameterized_transformers).to_csv(os.path.join(save_dir, 'reparameterized_transformers.csv'), index=False)