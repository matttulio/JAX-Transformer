import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
    context_window = 3
    vocab_size = round(sequence_length * 7.8125)
    vocab = list(range(vocab_size))
    embedding_dim = 50
    embedding_path = 'Datasets/glove/glove.6B.50d.txt'
    embedding_model = 'glove.6B.50d'
    seed = 42
    distr_param = 2
    n_classes = 2

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


results = pd.read_csv(os.path.join(retrieve_dir, 'frozen_transformer_result.csv'))
results.train_losses = results.train_losses.apply(ast.literal_eval)
results.val_losses = results.val_losses.apply(ast.literal_eval)
results.val_acc = results.val_acc.apply(ast.literal_eval)


for model_type, g in results.groupby('model_type'):
    acc = []
    for i,row in g.iterrows():
        plt.plot(row['val_acc'], c='orange')
        acc.append(row['val_acc'][-1])
    print(model_type, np.mean(acc), np.std(acc))
    plt.plot([], [], c='orange', label='validation')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title(model_type)
    plt.savefig(os.path.join(save_dir, f'accuracy_{model_type}.pdf'))
    plt.show()


for model_type, g in results.groupby('model_type'):
    for i,row in g.iterrows():
        plt.plot(row['train_losses'],c='blue')
        plt.plot(row['val_losses'],c='orange')
        plt.plot([],[],c='blue',label='train')
        plt.plot([],[],c='orange',label='validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title(model_type)
    plt.savefig(os.path.join(save_dir, f'loss_{model_type}.pdf'))
    plt.show()


def highlight_cell(x,y,color,transformer,ax):
    # Given a coordinate (x,y), highlight the corresponding cell using a colored frame in the ac
    # after having called imshow already
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False,color=color,lw=2)
    ax.add_patch(rect)
    return rect

def visualize_attention_matrix(x, transformer, ax, cmap='tab20b'):

    x_ = jnp.array(x)

    _, attn_probs = transformer.apply(transformer.params, x_)
    data = attn_probs[0]
    #data = A.detach().cpu().numpy()[0]
    ax.imshow(data,vmin=0,vmax=1.0,cmap=cmap)
    for i, x_i in enumerate(x):
        for j, x_j in enumerate(x):
            if x_i == x_j:
                highlight_cell(i,j, color='red',transformer=transformer,ax=ax)
    for k in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, k, f'{int(np.round(data[k, j]*100))}', ha='center', va='center', color='white')
    alpha = 'ABCDEFGHIJKLMNOPQRSTUVW'
    ax.set_xticks(np.arange(len(x)), [alpha[a] for a in x],fontsize=13)
    ax.set_yticks(np.arange(len(x)), [alpha[a] for a in x],fontsize=13)
    ax.tick_params(axis='x', which='both', bottom=False, top=True)
    ax.xaxis.tick_top()


# Extract colors from tab20b colormap
tab20b_colors = plt.cm.tab20b.colors

# Select specific colors from tab20b for your custom colormap
selected_colors = [tab20b_colors[2], tab20b_colors[6], tab20b_colors[10], tab20b_colors[14]]
selected_colors = tab20b_colors[1::2]

# Create a ListedColormap using the selected colors
custom_cmap = ListedColormap(selected_colors)


xs = [[1,1,2,2,3,3,1,1,1,1],
      [11,12,3,12,12,12,1,7,7,1],
      [14,13,12,11,10,9,8,7,6,5]]

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
plt.show()


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
    plt.savefig(os.path.join(save_dir,f'../figures/run_{r}_training_comparison_{orig_trans.attention_input}.pdf'),bbox_inches='tight')
    plt.show()



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
df = []
for model_type, g in results.groupby('model_type'):
  for i, row in g.iterrows():
    r = row['run']
    
    transformer_frozen_init = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, model_type)
    with open(os.path.join(retrieve_dir, f'run_{r}_initmodel_{model_type}_orig.pkl'), "rb") as file:
        init_orig_state = cloudpickle.load(file)
    init_orig_state = reparameterize(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, init_orig_state, model_type, rng)
    transformer_frozen_init.params = init_orig_state.params
    

    transformer_frozen = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, model_type)
    with open(os.path.join(retrieve_dir, f'run_{r}_model_{model_type}_orig.pkl'), "rb") as file:
        state = cloudpickle.load(file)
    state = reparameterize(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, state, model_type, rng)
    transformer_frozen.params = state.params
    
    reparam_trans = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, 'both')
    with open(os.path.join(retrieve_dir, f'run_{r}_model_{model_type}_retrained.pkl'), "rb") as file:
        state_rep = cloudpickle.load(file)
    state_rep = reparameterize(vocab_size, model_dim, hidden_dimension_fc, n_classes, sequence_length, state_rep, model_type, rng)
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