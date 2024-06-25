import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import pandas as pd
import ast
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.transformer import *
import cloudpickle
import jax.numpy as jnp
from flax.traverse_util import flatten_dict



#case_study = 1  # Plot results for primitive NLP dataset for next token prediction
case_study = 2   # Plot results for primitive NLP dataset for summing task
#case_study = 3  # Plot results for Next Histogram Task dataset

print("\n")

if(case_study == 1):

    num_samples = 50000
    sequence_length = 10
    context_window = 3
    vocab_size = round(sequence_length * 7.8125)
    vocab = list(range(vocab_size))
    embedding_dim = 50
    embedding_path = 'Datasets/glove/glove.6B.50d.txt'
    embedding_model = 'glove.6B.50d'
    seed = 42
    distr_param = 2
    n_classes = vocab_size + 1

    rs = np.random.RandomState(seed)
    rs.shuffle(vocab)

    print("Plotting results for primitive NLP dataset for next token prediction...")
    print(f"The parameters of the dataset are: num_samples={num_samples}, sequence_lenght={sequence_length}, context_window={context_window}")
    print(f"vocab_size={vocab_size}, embedding_dim={embedding_dim}, embedding_type={embedding_model}, seed={seed}, distribution_parameter={distr_param}\n")


    save_dir = f"Empirics/primitive_NLP_NTP_dataset_n_smpl{num_samples}__seq_len{sequence_length}__cont_win{context_window}__" \
    + f"v_size{vocab_size}__emb_dim{embedding_dim}__emb_type{embedding_model}__seed{seed}__d_par{distr_param}/figures"
    retrieve_dir = f"Empirics/primitive_NLP_NTP_dataset_n_smpl{num_samples}__seq_len{sequence_length}__cont_win{context_window}__" \
    + f"v_size{vocab_size}__emb_dim{embedding_dim}__emb_type{embedding_model}__seed{seed}__d_par{distr_param}"

elif(case_study == 2):

    num_samples = 50000
    sequence_length = 10
    context_window = 10
    vocab_size = round(sequence_length * 7.8125)
    vocab = list(range(vocab_size))
    embedding_dim = 50
    embedding_path = 'Datasets/glove/glove.6B.50d.txt'
    embedding_model = 'glove.6B.50d'
    seed = 42
    distr_param = 1.1
    n_classes = 2

    rs = np.random.RandomState(seed)
    rs.shuffle(vocab)

    print("Plotting result for primitive NLP dataset for next token prediction...")
    print(f"The parameters of the dataset are: num_samples={num_samples}, sequence_lenght={sequence_length}, context_window={context_window}")
    print(f"vocab_size={vocab_size}, embedding_dim={embedding_dim}, embedding_type={embedding_model}, seed={seed}, distribution_parameter={distr_param}\n")

    save_dir = f"Empirics/primitive_NLP_dataset_n_smpl{num_samples}__seq_len{sequence_length}__cont_win{context_window}__" \
    + f"v_size{vocab_size}__emb_dim{embedding_dim}__emb_type{embedding_model}__seed{seed}__d_par{distr_param}/figures"
    retrieve_dir = f"Empirics/primitive_NLP_dataset_n_smpl{num_samples}__seq_len{sequence_length}__cont_win{context_window}__" \
    + f"v_size{vocab_size}__emb_dim{embedding_dim}__emb_type{embedding_model}__seed{seed}__d_par{distr_param}"


elif(case_study == 3):

    num_samples = 50000
    sequence_length = 10
    vocab_size = 15
    seed = 42
    n_classes = 7

    print("Plotting result for Next Histogram Dataset...")
    print(f"The parameters of the datasets are: num_samples={num_samples}, sequence_lenght={sequence_length}, vocab_size={vocab_size}, seed={seed}\n")

    save_dir = f'Empirics/NextHistogramDataset_n_smpl{num_samples}__seq_len{sequence_length}__v_size{vocab_size}__seed{seed}/figures'
    retrieve_dir = f'Empirics/NextHistogramDataset_n_smpl{num_samples}__seq_len{sequence_length}__v_size{vocab_size}__seed{seed}'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


model_types = ['only_pos','only_sem']
hidden_dimension_fc = 128
model_dim = 64
select_run = 0

# ============== Results frozen transformer ============== #
results = pd.read_csv(os.path.join(retrieve_dir, 'frozen_transformer_result.csv'))
results.train_losses = results.train_losses.apply(ast.literal_eval)
results.val_losses = results.val_losses.apply(ast.literal_eval)
results.val_acc = results.val_acc.apply(ast.literal_eval)

n_runs = np.max(results['run']) + 1

# ============== Validation accuracy vs epochs ============== #
idx = 0
num_colors = n_runs
colors = plt.colormaps['viridis'].resampled(num_colors)  # You can change 'tab10' to other colormaps

for model_type, g in results.groupby('model_type'):
    acc = []
    for i,row in g.iterrows():
        idx += 1
        color = colors(idx % num_colors)
        plt.plot(row['val_acc'], color=color, label=f'Validation Acc Run {idx}')
        acc.append(row['val_acc'][-1])
    print(model_type, np.mean(acc), np.std(acc))

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title(model_type)
    plt.savefig(os.path.join(save_dir, f'accuracy_{model_type}.pdf'))
    plt.clf()
    plt.close()
    #plt.show()


# ============== Val and Train loss against epochs ============== #
idx = 0
for model_type, g in results.groupby('model_type'):
    fig, ax = plt.subplots()
    for i, row in g.iterrows():
        idx += 1
        color = colors(idx % num_colors)
        ax.plot(row['train_losses'], color=color, label=f'Train Run {idx}')
        ax.plot(row['val_losses'], color=color, label=f'Val Run {idx}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title(model_type)
    plt.savefig(os.path.join(save_dir, f'loss_{model_type}.pdf'))
    plt.clf()
    plt.close()
    #plt.show()


cmap_diff = LinearSegmentedColormap.from_list("grad_cmap", ["cornflowerblue", "lightgray", "#d15f90"])

idx = 0

for run in results['run'].unique():
    fig, ax = plt.subplots()

    # Filter for only_pos and only_sem for the current run
    only_pos_val_loss = results[(results['run'] == run) & (results['model_type'] == 'only_pos')]['val_losses'].values[0]
    only_sem_val_loss = results[(results['run'] == run) & (results['model_type'] == 'only_sem')]['val_losses'].values[0]
    
    # Calculate the difference between the accuracies
    val_loss_diff = [pos - sem for pos, sem in zip(only_pos_val_loss, only_sem_val_loss)]
    
    limit = np.max((np.abs(np.max(val_loss_diff)), np.abs(np.min(val_loss_diff))))
    #limit = 0.001
    norm = Normalize(vmin=-1*limit, vmax=limit)
    colors_diff = [cmap_diff(norm(val)) for val in val_loss_diff]

    ax.plot(range(len(val_loss_diff)), val_loss_diff, linestyle='-', linewidth=1, color='gray')
    ax.scatter(range(len(val_loss_diff)), val_loss_diff, s=50, c=colors_diff)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Validation Loss Difference') 
    #ax.legend()
    ax.set_title('Difference Validation Losses')
    plt.savefig(os.path.join(save_dir, f'val_loss_difference_{model_types[0]}_{model_types[1]}_run_{run}.pdf'))
    plt.clf()
    plt.close()

# ============== Attention map ============== #
def highlight_cell(x, y, color, ax):
    # Given a coordinate (x,y), highlight the corresponding cell using a colored frame in the ac
    # after having called imshow already
    rect = plt.Rectangle((x-.5, y-.5), 1, 1, fill=False, color=color, lw=2)
    ax.add_patch(rect)
    return rect

def visualize_attention_matrix(x, transformer, ax, cmap='tab20b'):

    x_ = jnp.array(x)

    _, attn_probs = transformer.apply(transformer.params, x_)
    data = attn_probs[0]
    #data = A.detach().cpu().numpy()[0]
    ax.imshow(data,vmin=0,vmax=1.0,cmap=cmap)

    if(case_study == 3):
        for i, x_i in enumerate(x):
            for j, x_j in enumerate(x):
                if x_i == x_j:
                    highlight_cell(i, j, color='red', ax=ax)
        for k in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, k, f'{int(np.round(data[k, j]*100))}', ha='center', va='center', color='white')

        alpha = 'ABCDEFGHIJKLMNOPQRSTUVW'
        ax.set_xticks(np.arange(len(x)), [alpha[a] for a in x], fontsize=13)
        ax.set_yticks(np.arange(len(x)), [alpha[a] for a in x], fontsize=13)

        ax.tick_params(axis='x', which='both', bottom=False, top=True)
        ax.xaxis.tick_top()


    elif(case_study == 1 or case_study == 2):

        if(embedding_path == None):
            # Build a random embedding
            embeddings = nn.Embedding(vocab_size, embedding_dim)  # embedding layer
            embedding_matrix = embeddings.weight.data  # embedding matrix
        else:
            # Load pre-trained embedding matrix
            embedding_vectors = []
            with open(embedding_path, 'r', encoding = 'utf-8') as f:
                next(f)  # Skip the header or first line if any
                # Use the readlines() method to read all lines into a list
                lines = f.readlines()

                # Count the number of lines in the list
                num_rows = len(lines)

                step = num_rows // vocab_size
                for i, line in enumerate(lines):
                    if i >= vocab_size * step:  # Break if enough vectors are read
                        break
                    if i % step == 0:  # Only take every step-th vector
                        values = line.split()
                        vector = torch.tensor([float(val) for val in values[1:]])
                        embedding_vectors.append(vector)
                    
            embedding_matrix = embedding_vectors

        max_pos = sequence_length
        position_enc = torch.tensor([[torch.sin(torch.tensor(pos / (10000 ** (i // 2 * 2.0 / embedding_dim)), dtype=torch.float)) if i % 2 == 0 else torch.cos(torch.tensor(pos / (10000 ** (i // 2 * 2.0 / embedding_dim)), dtype=torch.float)) for i in range(embedding_dim)] for pos in range(max_pos)], dtype=torch.float)

        highlight_cell(0, 0, color='white', ax=ax)
        for i in range(sequence_length-2, 0, -1):

            token_index = vocab.index(x[i+1])
            token_embedding = embedding_matrix[token_index] + position_enc[i+1]

            similarities = [torch.dot(embedding_matrix[vocab.index(x[k])] + position_enc[k], token_embedding) for k in range(i)]
            similarities = torch.tensor([similarities])
            _, next_token_i = torch.topk(similarities, i)

            if(next_token_i[0][0] == next_token_i[0][-1]):
                highlight_cell(next_token_i[0][0], i, color='white', ax=ax)
            else:
                highlight_cell(next_token_i[0][0], i, color='green', ax=ax)
                highlight_cell(next_token_i[0][-1], i, color='red', ax=ax)

        for k in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, k, f'{int(np.round(data[k, j]*100))}', ha='center', va='center', color='white')

        ax.set_xticks(np.arange(len(x)), [vocab[a] for a in x], fontsize=7)
        ax.set_yticks(np.arange(len(x)), [vocab[a] for a in x], fontsize=7)
        ax.tick_params(axis='x', which='both', bottom=False, top=True)
        ax.xaxis.tick_top()

# Extract colors from tab20b colormap
tab20b_colors = plt.cm.tab20b.colors

# Select specific colors from tab20b for your custom colormap
selected_colors = [tab20b_colors[2], tab20b_colors[6], tab20b_colors[10], tab20b_colors[14]]
selected_colors = tab20b_colors[1::2]

# Create a ListedColormap using the selected colors
custom_cmap = ListedColormap(selected_colors)

with open(os.path.join(retrieve_dir, 'input_sequences.pkl'), "rb") as file:
       input_sequences = cloudpickle.load(file)

xs = [input_sequences[0][0].tolist(), input_sequences[0][1].tolist(), input_sequences[0][2].tolist()]

print(f"n_classes = ", n_classes)

# visualize the first run
transformer_only_pos = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, 'only_pos', True)
with open(os.path.join(retrieve_dir, f'run_{select_run}_model_only_pos_orig.pkl'), "rb") as file:
    state_only_pos = cloudpickle.load(file)
transformer_only_pos.params = state_only_pos.params

transformer_only_sem = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, 'only_sem', True)
with open(os.path.join(retrieve_dir, f'run_{select_run}_model_only_sem_orig.pkl'), "rb") as file:
    state_only_sem = cloudpickle.load(file)
transformer_only_sem.params = state_only_sem.params

transformers = [
    transformer_only_pos,
    transformer_only_sem,
]


fig, axes = plt.subplots(figsize=(15,8),ncols=3,nrows=2)
cmap = custom_cmap# 'Paired'
data = np.random.random((10, 10))
im1 = axes[0,0].imshow(data, cmap=cmap, vmin=0, vmax=1.0)
for i, transformer in enumerate(transformers):
  axes[i,0].set_ylabel("Positional" if transformer.attention_input == 'only_pos' else "Semantic", fontsize=14)
  print(transformer.attention_input)
  for j, x in enumerate(xs):
    axes[0,j].set_title(f"Example Sequence #{j+1}",fontsize=10)
    visualize_attention_matrix(x, transformer, axes[i,j], cmap=cmap)


norm = Normalize(vmin=0, vmax=1.0)
cbar = fig.colorbar(im1, ax=axes, norm=norm)
cbar.set_label('attention value',fontsize=12)
plt.savefig(os.path.join(save_dir, f'tiny_example.pdf'))
plt.clf()
plt.close()
#plt.show()



# ============== Results for reparam transformer ============== #

for model_type, g in results.groupby('model_type'):
  for i, row in g.iterrows():
    r = row['run']
    print(r)
    
    orig_trans = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, model_type, True)
    with open(os.path.join(retrieve_dir, f'run_{r}_model_{model_type}_orig.pkl'), "rb") as file:
        state_orig = cloudpickle.load(file)
    orig_trans.params = state_orig.params
    print(os.path.join(save_dir, f'run_{r}_model_{model_type}_orig.pkl'))
    
    reparam_trans = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, 'both', True)
    with open(os.path.join(retrieve_dir, f'run_{r}_model_{model_type}_retrained.pkl'), "rb") as file:
        state_rep = cloudpickle.load(file)
    reparam_trans.params = state_rep.params
    print(os.path.join(save_dir, f'run_{r}_model_{model_type}_retrained.pkl'))
    
    transformers = [
      orig_trans,
      reparam_trans,
    ]
    
    # ============== Attention maps retrained ============== #
    data = np.random.random((10, 10))
    fig, axes = plt.subplots(figsize=(15,8),ncols=3,nrows=2)
    im1 = axes[0,0].imshow(data, cmap=cmap, vmin=0, vmax=1.0)
    for i, transformer in enumerate(transformers):
      print(transformer.attention_input)
      for j, x in enumerate(xs):
        axes[0,j].set_title(f"Example Sequence #{j+1}",fontsize=10)
        visualize_attention_matrix(x, transformer, axes[i,j], cmap=cmap)

    axes[0,0].set_ylabel(r'$\theta_{sem}$' if orig_trans.attention_input == 'only_sem' else r'$\theta_{pos}$')
    axes[1,0].set_ylabel(r'$\tilde{\theta}_{sem}$' if orig_trans.attention_input == 'only_sem' else r'$\tilde{\theta}_{pos}$')

    norm = Normalize(vmin=0, vmax=1.0)
    cbar = fig.colorbar(im1, ax=axes, norm=norm)
    cbar.set_label('attention value',fontsize=12)
    plt.savefig(os.path.join(save_dir, f'run_{r}_training_comparison_{orig_trans.attention_input}.pdf'),bbox_inches='tight')
    plt.clf()
    plt.close()
    #plt.show()


# ============== Distance between the models's parameters ============== #
def model_distance(model1, model2, only_zeros=False):
    params1 = [param for param in model1.parameters()]
    params2 = [param for param in model2.parameters()]

    distance = 0.0
    for p1, p2 in zip(params1, params2):
        if only_zeros:
          mask = p2.flatten() == 0.0
          distance += torch.norm(p1.flatten()[mask] - p2.flatten()[mask], 2)
        else:
          distance += torch.norm(p1 - p2, 2)

    return distance.item()


def model_distance(model1, model2, only_zeros=False):
    distance = 0.0

    params1 = model1.params['params']
    params2 = model2.params['params']

    # Flatten the parameter dictionaries
    flat_params1 = flatten_dict(params1)
    flat_params2 = flatten_dict(params2)

    # Ensure both parameter lists have the same structure
    assert flat_params1.keys() == flat_params2.keys(), "Parameter structures do not match."

    # Iterate through the flattened parameters
    for key in flat_params1.keys():
        p1 = flat_params1[key]
        p2 = flat_params2[key]

        if isinstance(p1, jnp.ndarray) and isinstance(p2, jnp.ndarray):
            if only_zeros:
                mask = p2 == 0.0
                if jnp.any(mask):
                    distance += jnp.linalg.norm(p1[mask] - p2[mask])
            else:
                distance += jnp.linalg.norm(p1 - p2)

    return distance

seed = 42
rng = random.PRNGKey(seed)
batch_size = 32
learning_rate = 1e-4
dummy_input = np.ones(shape=(batch_size, sequence_length), dtype=np.int8)

df = []
for model_type, g in results.groupby('model_type'):
  for i, row in g.iterrows():
    r = row['run']
    
    transformer_frozen_init = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, model_type)
    with open(os.path.join(retrieve_dir, f'run_{r}_initmodel_{model_type}_orig.pkl'), "rb") as file:
        init_orig_state = cloudpickle.load(file)
    init_orig_state = reparameterize(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, init_orig_state, model_type, dummy_input, learning_rate, rng)
    transformer_frozen_init.params = init_orig_state.params
    

    transformer_frozen = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, model_type)
    with open(os.path.join(retrieve_dir, f'run_{r}_model_{model_type}_orig.pkl'), "rb") as file:
        state = cloudpickle.load(file)
    state = reparameterize(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, state, model_type, dummy_input, learning_rate, rng)
    transformer_frozen.params = state.params
    
    reparam_trans = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, 'both')
    with open(os.path.join(retrieve_dir, f'run_{r}_model_{model_type}_retrained.pkl'), "rb") as file:
        state_rep = cloudpickle.load(file)
    state_rep = reparameterize(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, state_rep, model_type, dummy_input, learning_rate, rng)
    reparam_trans.params = state_rep.params
        
    dist_frozen_to_SGD = model_distance(reparam_trans, transformer_frozen) 
    dist_frozen_to_SGD_zeros = model_distance(reparam_trans,transformer_frozen,only_zeros=True)
    #dist_frozen_to_SGD_zeros = model_distance(transformer_frozen,reparam_trans,only_zeros=True)
    dist_init_to_frozen = model_distance(transformer_frozen_init, transformer_frozen)
        
    df.append({
        'distance_frozen_to_SGD': dist_frozen_to_SGD,
        'distance_frozen_to_SGD_only_zeros': dist_frozen_to_SGD_zeros,
        'distance_init_to_frozen': dist_init_to_frozen,
        'model_type': model_type,
        'run': r
    })

df = pd.DataFrame(df)
df = df.groupby('model_type').agg(['mean','std'])
print(df)

# ============== Distribution of predicted tokens ============== #

if(case_study == 1):

    with open(os.path.join(retrieve_dir, 'sequences_to_predict.pkl'), "rb") as file:
       sequences_to_predict = np.asarray(np.concatenate(cloudpickle.load(file)))
    
    distr_to_pred = [] 
    for i in (vocab):
        z = len(sequences_to_predict[np.where(sequences_to_predict == i)])
        distr_to_pred.append(z)

    # Compute the frequencies of each token
    num_tok = sequences_to_predict.size
    distr_to_pred = np.array(distr_to_pred) / num_tok

    # Sort the frequencies in descending order
    distr_to_pred[::-1].sort()
    distr_to_pred = distr_to_pred.tolist()
    width = 0.4
    

    for i in range(n_runs):
        for model_type in model_types:
            with open(os.path.join(retrieve_dir, f'predicted_sequences_run_{i}_model_{model_type}.pkl'), "rb") as file:
                predicted_sequences = np.asarray(np.concatenate(cloudpickle.load(file)))

           
            distr_predicted = [] 
            for j in (vocab):
                z = len(predicted_sequences[np.where(predicted_sequences == j)])
                distr_predicted.append(z)

            # Compute the frequencies of each token
            num_tok = sequences_to_predict.size
            distr_predicted = np.array(distr_predicted) / num_tok

            # Sort the frequencies in descending order
            distr_predicted[::-1].sort()
            distr_predicted = distr_predicted.tolist()

            plt.bar(np.arange(vocab_size) - width/2, distr_to_pred, width, color=colors(2 % num_colors), label = 'Distribution tokens to predict')
            plt.bar(np.arange(vocab_size) + width/2, distr_predicted, width, color=colors(0 % num_colors), label=f'Predicted distribution (Run {i} Model {model_type})')
            plt.xlabel('Token Rank')
            plt.ylabel('Frequency')
            plt.title('Input vs Output Token Frequency Distributions')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'distr_predict_vs_label_run_{i}_{model_type}.pdf'))
            plt.clf()
            plt.close()


results = pd.read_csv(os.path.join(retrieve_dir, 'reparameterized_transformers.csv'))
results.train_losses = results.train_losses.apply(ast.literal_eval)
results.val_losses = results.val_losses.apply(ast.literal_eval)
results.val_acc = results.val_acc.apply(ast.literal_eval)

# ============== Accuracy reparametrized model ============== #
idx = 0
num_colors = np.max(results['run']) + 1
colors = plt.colormaps['viridis'].resampled(num_colors)  # You can change 'tab10' to other colormaps


for model_type, g in results.groupby('model_type'):
    acc = []
    for i, row in g.iterrows():
        idx += 1
        color = colors(idx % num_colors)
        plt.plot(row['val_acc'], color=color, label=f'Validation Acc Run{idx}')
        acc.append(row['val_acc'][-1])
    print(model_type, np.mean(acc), np.std(acc))

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f"Reparametrized {model_type}")
    plt.savefig(os.path.join(save_dir, f'accuracy_reparametrized_{model_type}.pdf'))
    plt.clf()
    plt.close()
    #plt.show()


# ============== Train and Val loss reparametrized model ============== #
idx = 0
for model_type, g in results.groupby('model_type'):
    fig, ax = plt.subplots()
    for i, row in g.iterrows():
        idx += 1
        color = colors(idx % num_colors)
        ax.plot(row['train_losses'], color=color, label=f'Train Run {idx}')
        ax.plot(row['val_losses'], color=color, label=f'Val Run {idx}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title(f"Reparametrized {model_type}")
    plt.savefig(os.path.join(save_dir, f'loss_reparametrized_{model_type}.pdf'))
    plt.clf()
    plt.close()
    #plt.show()