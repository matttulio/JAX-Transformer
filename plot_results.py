import numpy as np
import matplotlib.pyplot as plt
import scienceplots
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

plt.style.use('science')

case_study = 1  # Plot results for primitive NLP dataset for next token prediction
#case_study = 2   # Plot results for primitive NLP dataset for summing task
#case_study = 3  # Plot results for Next Histogram Task dataset

print("\n")

if(case_study == 1):

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
    temperature = 2
    n_classes = vocab_size + 1

    rs = np.random.RandomState(seed)
    rs.shuffle(vocab)

    print("Plotting results for primitive NLP dataset for next token prediction...")
    print(f"The parameters of the dataset are: num_samples={num_samples}, sequence_lenght={sequence_length}, context_window={context_window}")
    print(f"vocab_size={vocab_size}, embedding_dim={embedding_dim}, embedding_type={embedding_model}, seed={seed}, distribution_parameter={distr_param}, temperature={temperature}\n")


    save_dir = f"Empirics/primitive_NLP_NTP_dataset_n_smpl{num_samples}__seq_len{sequence_length}__cont_win{context_window}__" \
    + f"v_size{vocab_size}__emb_dim{embedding_dim}__emb_type{embedding_model}__seed{seed}__d_par{distr_param}__temp{temperature}/figures"
    retrieve_dir = f"Empirics/primitive_NLP_NTP_dataset_n_smpl{num_samples}__seq_len{sequence_length}__cont_win{context_window}__" \
    + f"v_size{vocab_size}__emb_dim{embedding_dim}__emb_type{embedding_model}__seed{seed}__d_par{distr_param}__temp{temperature}"

elif(case_study == 2):

    num_samples = 50000
    sequence_length = 10
    context_window = 3
    vocab_size = round(sequence_length * 7.8125)
    vocab = list(range(vocab_size))
    embedding_dim = 50
    embedding_path = 'Datasets/glove/glove.6B.50d.txt'
    embedding_model = 'glove.6B.50d'
    seed = 42
    distr_param = 1.1
    temperature = 2
    n_classes = 2

    rs = np.random.RandomState(seed)
    rs.shuffle(vocab)

    print("Plotting result for primitive NLP dataset for next token prediction...")
    print(f"The parameters of the dataset are: num_samples={num_samples}, sequence_lenght={sequence_length}, context_window={context_window}")
    print(f"vocab_size={vocab_size}, embedding_dim={embedding_dim}, embedding_type={embedding_model}, seed={seed}, distribution_parameter={distr_param}, temperature={temperature}\n")

    save_dir = f"Empirics/primitive_NLP_dataset_n_smpl{num_samples}__seq_len{sequence_length}__cont_win{context_window}__" \
    + f"v_size{vocab_size}__emb_dim{embedding_dim}__emb_type{embedding_model}__seed{seed}__d_par{distr_param}__temp{temperature}/figures"
    retrieve_dir = f"Empirics/primitive_NLP_dataset_n_smpl{num_samples}__seq_len{sequence_length}__cont_win{context_window}__" \
    + f"v_size{vocab_size}__emb_dim{embedding_dim}__emb_type{embedding_model}__seed{seed}__d_par{distr_param}__temp{temperature}"


elif(case_study == 3):

    num_samples = 200000
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
train_steps = list(range(0, len(results['train_losses'][0])))
val_steps = list(range(0, (len(results['train_losses'][0]) - (len(results['train_losses'][0]) % 10)) + 1, 10))
if((len(results['train_losses'][0]) - 1) % 10 != 0):
    val_steps.append(len(results['train_losses'][0]) - 1)
val_steps.insert(1, 1)

# print(len(results['train_losses'][0]), len(results['val_losses'][0]))
# print(train_steps)
# print(val_steps)
# exit()
# ============== Validation accuracy vs steps ============== #
idx = 0
num_colors = n_runs
colors = plt.colormaps['viridis'].resampled(num_colors)  # You can change 'tab10' to other colormaps

for model_type, g in results.groupby('model_type'):
    all_acc = []
    for i, row in g.iterrows():
        idx += 1
        all_acc.append(row['val_acc'])
    
    # Calculate mean and standard deviation of accuracies
    mean_acc = np.mean(all_acc, axis=0)
    std_acc = np.std(all_acc, axis=0)

    color = colors(idx % num_colors)
    plt.figure(figsize=(10, 8))
    plt.plot(val_steps, mean_acc, color=color, label=f'Mean Validation Acc')
    plt.fill_between(val_steps, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.2)
    #plt.axhline(y=0.8919672924086934, color='r', linestyle='--', label='Baseline')

    print(model_type, np.mean(mean_acc), np.std(mean_acc))

    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend(fontsize=14)
    plt.title("Only Positional" if model_type == "only_pos" else "Only Semantic")
    plt.savefig(os.path.join(save_dir, f'accuracy_{model_type}.pdf'))
    plt.clf()
    plt.close()
    #plt.show()
#exit()


# ============== Val and Train loss against steps ============== #
idx = 0
for model_type, g in results.groupby('model_type'):
    all_train_losses = []
    all_val_losses = []
    
    for i, row in g.iterrows():
        idx += 1
        all_train_losses.append(row['train_losses'])
        all_val_losses.append(row['val_losses'])
    
    # Calculate mean and standard deviation of losses
    mean_train_losses = np.mean(all_train_losses, axis=0)
    std_train_losses = np.std(all_train_losses, axis=0)
    mean_val_losses = np.mean(all_val_losses, axis=0)
    std_val_losses = np.std(all_val_losses, axis=0)

    
    fig, ax = plt.subplots(figsize=(10, 8))
    color = colors(idx % num_colors)
    # Plot mean and std for train losses
    ax.plot(train_steps, mean_train_losses, color=color, label='Mean Train Loss')
    ax.fill_between(train_steps, mean_train_losses - std_train_losses, mean_train_losses + std_train_losses, color=color, alpha=0.2)
    
    # Plot mean and std for validation losses
    ax.plot(val_steps, mean_val_losses, color=colors((idx + 2) % num_colors), label='Mean Val Loss')
    ax.fill_between(val_steps, mean_val_losses - std_val_losses, mean_val_losses + std_val_losses, color=colors((idx + 2) % num_colors), alpha=0.2)
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=14)
    ax.set_title("Only Positional" if model_type == "only_pos" else "Only Semantic")
    plt.savefig(os.path.join(save_dir, f'loss_{model_type}.pdf'))
    plt.clf()
    plt.close()
    #plt.show()

cmap_diff = LinearSegmentedColormap.from_list("grad_cmap", ["cornflowerblue", "lightgray", "#d15f90"])

idx = 0

all_val_loss_diffs = []

for run in results['run'].unique():
    # Filter for only_pos and only_sem for the current run
    only_pos_val_loss = results[(results['run'] == run) & (results['model_type'] == 'only_pos')]['val_losses'].values
    only_sem_val_loss = results[(results['run'] == run) & (results['model_type'] == 'only_sem')]['val_losses'].values
    
    if len(only_pos_val_loss) > 0 and len(only_sem_val_loss) > 0:
        only_pos_val_loss = only_pos_val_loss[0]
        only_sem_val_loss = only_sem_val_loss[0]

        # Calculate the difference between the accuracies
        val_loss_diff = [pos - sem for pos, sem in zip(only_pos_val_loss, only_sem_val_loss)]
        all_val_loss_diffs.append(val_loss_diff)

# Calculate mean and standard deviation of differences
mean_val_loss_diff = np.mean(all_val_loss_diffs, axis=0)
std_val_loss_diff = np.std(all_val_loss_diffs, axis=0)

# Normalize differences for coloring
limit = np.min((np.abs(np.max(mean_val_loss_diff)), np.abs(np.min(mean_val_loss_diff))))
norm = Normalize(vmin=-0.001 * limit, vmax=0.001*limit)
colors_diff = [cmap_diff(norm(val)) for val in mean_val_loss_diff]

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(val_steps, mean_val_loss_diff, linestyle='-', linewidth=1, color='gray')
ax.fill_between(val_steps, mean_val_loss_diff - std_val_loss_diff, mean_val_loss_diff + std_val_loss_diff, color='blueviolet', alpha=0.1)
ax.scatter(val_steps, mean_val_loss_diff, s=50, c=colors_diff)

ax.set_xlabel('Steps')
ax.set_ylabel('Validation Loss Difference')
ax.set_title('Mean Difference in Validation Losses')
#ax.set_xlim([0.6, 15])
#ax.set_ylim([-0.0001, 0.0001])
plt.savefig(os.path.join(save_dir, 'mean_val_loss_difference_only_pos_only_sem.pdf'))
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
                #num_rows = len(lines)

                #step = num_rows // vocab_size
                step = 1
                for i, line in enumerate(lines):
                    if i >= vocab_size * step:  # Break if enough vectors are read
                        break
                    if i % step == 0:  # Only take every step-th vector
                        values = line.split()
                        vector = torch.tensor([float(val) for val in values[1:]])
                        embedding_vectors.append(vector)
                    
            embedding_matrix = embedding_vectors

        max_pos = context_window
        position_enc = torch.tensor([[1 - (0.5 * pos / (max_pos - 1)) for _ in range(embedding_dim)] for pos in range(max_pos - 1, -1, -1)], dtype=torch.float)
        
        highlight_cell(0, 0, color='white', ax=ax)
        for length in range(sequence_length-1, 0, -1):
           
            token_index = vocab.index(x[length])
            token_embedding = embedding_matrix[token_index]

            j = context_window-1
            similarities = []
            for i in range(length-1, max(length - context_window - 1, -1), -1):
                token_index = vocab.index(x[i])
                similarities.append(torch.dot(embedding_matrix[token_index] * position_enc[j], token_embedding).item())
                j -= 1

            #similarities = [torch.dot(torch.matmul(embedding_matrix[vocab.index(x[k])], position_enc[k - max(i-context_window, 0)]), token_embedding) for k in range(i, max(i-context_window, 0), -1)]
            similarities = torch.tensor(similarities)

            _, most_similar_tokens_i = torch.topk(similarities, min(context_window, length))

            if(most_similar_tokens_i[0] == most_similar_tokens_i[-1]):
                highlight_cell(length - 1 - most_similar_tokens_i[0], length-1, color='white', ax=ax)
            else:
                highlight_cell(length - 1 - most_similar_tokens_i[0], length-1, color='green', ax=ax)
                highlight_cell(length - 1 - most_similar_tokens_i[-1], length-1, color='red', ax=ax)

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

plt.rcParams['xtick.major.size'] = 0
plt.rcParams['ytick.major.size'] = 0
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 0

fig, axes = plt.subplots(figsize=(15,8), ncols=3, nrows=2)
cmap = custom_cmap# 'Paired'
data = np.random.random((10, 10))
im1 = axes[0,0].imshow(data, cmap=cmap, vmin=0, vmax=1.0)
for i, transformer in enumerate(transformers):
  axes[i,0].set_ylabel("Positional" if transformer.attention_input == 'only_pos' else "Semantic", fontsize=14)
  print(transformer.attention_input)
  for j, x in enumerate(xs):
    axes[0,j].set_title(f"Example Sequence {j+1}",fontsize=10)
    visualize_attention_matrix(x, transformer, axes[i,j], cmap=cmap)

norm = Normalize(vmin=0, vmax=1.0)
cbar = fig.colorbar(im1, ax=axes, norm=norm)
cbar.set_label('Attention Value',fontsize=12)
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
    print(os.path.join(retrieve_dir, f'run_{r}_model_{model_type}_orig.pkl'))
    
    reparam_trans = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, 'both', True)
    with open(os.path.join(retrieve_dir, f'run_{r}_model_{model_type}_retrained.pkl'), "rb") as file:
        state_rep = cloudpickle.load(file)
    reparam_trans.params = state_rep.params
    print(os.path.join(retrieve_dir, f'run_{r}_model_{model_type}_retrained.pkl'))
    
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
        axes[0,j].set_title(f"Example Sequence {j+1}",fontsize=10)
        visualize_attention_matrix(x, transformer, axes[i,j], cmap=cmap)

    axes[0,0].set_ylabel(r'$\theta_{sem}$' if orig_trans.attention_input == 'only_sem' else r'$\theta_{pos}$')
    axes[1,0].set_ylabel(r'$\tilde{\theta}_{sem}$' if orig_trans.attention_input == 'only_sem' else r'$\tilde{\theta}_{pos}$')

    norm = Normalize(vmin=0, vmax=1.0)
    cbar = fig.colorbar(im1, ax=axes, norm=norm)
    cbar.set_label('Attention Value',fontsize=12)
    plt.savefig(os.path.join(save_dir, f'run_{r}_training_comparison_{orig_trans.attention_input}.pdf'),bbox_inches='tight')
    plt.clf()
    plt.close()
    #plt.show()

plt.rcdefaults()
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
    for i in vocab:
        z = len(sequences_to_predict[sequences_to_predict == i])
        distr_to_pred.append(z)

    num_tok = sequences_to_predict.size
    distr_to_pred = np.array(distr_to_pred) / num_tok
    distr_to_pred[::-1].sort()
    distr_to_pred = distr_to_pred.tolist()
    width = 0.4

    # Initialize a dictionary to collect distributions
    all_distr_predicted = {model_type: [] for model_type in model_types}

    for i in range(n_runs):
        for model_type in model_types:
            with open(os.path.join(retrieve_dir, f'predicted_sequences_run_{i}_model_{model_type}.pkl'), "rb") as file:
                predicted_sequences = np.asarray(np.concatenate(cloudpickle.load(file)))

            distr_predicted = []
            for j in vocab:
                z = len(predicted_sequences[predicted_sequences == j])
                distr_predicted.append(z)

            num_tok = sequences_to_predict.size
            distr_predicted = np.array(distr_predicted) / num_tok
            distr_predicted[::-1].sort()
            distr_predicted = distr_predicted.tolist()

            all_distr_predicted[model_type].append(distr_predicted)

    # Calculate mean and standard deviation for each model type
    mean_distr_predicted = {model_type: np.mean(all_distr_predicted[model_type], axis=0) for model_type in model_types}
    std_distr_predicted = {model_type: np.std(all_distr_predicted[model_type], axis=0) for model_type in model_types}

    # Plotting the distributions for each model type
    for model_type in model_types:
        fig, ax = plt.subplots(figsize=(10, 8))
        # Plot the true distribution
        ax.bar(np.arange(vocab_size) - width / 2, distr_to_pred, width, color=colors(2 % n_runs), label='Distribution tokens to predict')

        mean_values = mean_distr_predicted[model_type]
        std_values = std_distr_predicted[model_type]

        # Plot the predicted mean distribution
        ax.bar(np.arange(vocab_size) + width / 2, mean_values, width, yerr=std_values, capsize=5, color=colors(0 % n_runs), label=f'Mean Predicted Distribution ({model_type})')

        ax.set_xlabel('Token Rank')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Input vs Output Token Frequency Distributions ({model_type})')
        ax.legend(fontsize=14)

        plt.savefig(os.path.join(save_dir, f'mean_distr_predict_vs_label_{model_type}.pdf'))
        plt.clf()
        plt.close()



results = pd.read_csv(os.path.join(retrieve_dir, 'reparameterized_transformers.csv'))
results.train_losses = results.train_losses.apply(ast.literal_eval)
results.val_losses = results.val_losses.apply(ast.literal_eval)
results.val_acc = results.val_acc.apply(ast.literal_eval)

train_steps = list(range(0, len(results['train_losses'][0])))
val_steps = list(range(0, (len(results['train_losses'][0]) - (len(results['train_losses'][0]) % 10)) + 1, 10))
if((len(results['train_losses'][0]) - 1) % 10 != 0):
    val_steps.append(len(results['train_losses'][0]) - 1)
val_steps.insert(1, 1)

# ============== Accuracy reparametrized model ============== #
for model_type, g in results.groupby('model_type'):
    val_acc_all_runs = []

    for i, row in g.iterrows():
        val_acc_all_runs.append(row['val_acc'])

    val_acc_all_runs = np.array(val_acc_all_runs)
    mean_val_acc = np.mean(val_acc_all_runs, axis=0)
    std_val_acc = np.std(val_acc_all_runs, axis=0)

    color = colors(0 % num_colors)

    plt.figure(figsize=(10, 8))
    plt.plot(val_steps, mean_val_acc, color=color, label='Mean Validation Accuracy')
    plt.fill_between(val_steps, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, color=color, alpha=0.3)

    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend(fontsize=14)
    plt.title(f"Reparametrized {model_type}")
    plt.savefig(os.path.join(save_dir, f'accuracy_reparametrized_{model_type}.pdf'))
    plt.clf()
    plt.close()
    #plt.show()

# ============== Train and Val loss reparametrized model ============== #
for model_type, g in results.groupby('model_type'):
    train_losses_all_runs = []
    val_losses_all_runs = []

    for i, row in g.iterrows():
        train_losses_all_runs.append(row['train_losses'])
        val_losses_all_runs.append(row['val_losses'])

    train_losses_all_runs = np.array(train_losses_all_runs)
    val_losses_all_runs = np.array(val_losses_all_runs)
    mean_train_losses = np.mean(train_losses_all_runs, axis=0)
    std_train_losses = np.std(train_losses_all_runs, axis=0)
    mean_val_losses = np.mean(val_losses_all_runs, axis=0)
    std_val_losses = np.std(val_losses_all_runs, axis=0)


    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(train_steps, mean_train_losses, color=colors(idx % num_colors), label='Mean Training Loss')
    ax.fill_between(train_steps, mean_train_losses - std_train_losses, mean_train_losses + std_train_losses, color=colors(idx % num_colors), alpha=0.3)
    
    ax.plot(val_steps, mean_val_losses, color=colors((idx + 2) % num_colors), label='Mean Validation Loss')
    ax.fill_between(val_steps, mean_val_losses - std_val_losses, mean_val_losses + std_val_losses, color=colors((idx + 2) % num_colors), alpha=0.3)

    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=14)
    ax.set_title(f"Reparametrized {model_type}")
    plt.savefig(os.path.join(save_dir, f'loss_reparametrized_{model_type}.pdf'))
    plt.clf()
    plt.close()
    #plt.show()