import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset
import jax.numpy as jnp
from scipy.stats import moment, ortho_group
from tqdm.auto import tqdm
import os
import pickle
import scienceplots


class PrimitiveNLP(Dataset):
    def __init__(self, num_samples, sequence_length, context_window, vocab, embedding_matrix, seed = 42, distr_param = 1.1, temperature = 2):

        """
        Generate a dataset for next token prediction that emulates natural language.

        Args:
        - num_samples (int): Number of samples to generate.
        - sequence_length (int): Length of each sequence.
        - context_window (int): Lenght of the context.
        - vocab (list): Vocabulary.
        - embedding_dim (int): Dimension of the token embeddings.
        - embedding_path (string): Path for retrieving a pretreined embedding.
        - seed (int): Seed for reproducibility.
        - distr_param (float): Parameter of the zipf distribution.

        Returns:
        - X (numpy array): Input sequences with shape (num_samples, sequence_length).
        - y (numpy array): Target labels with shape (num_samples,).
        """

        # Set the seed for reproducibility
        rs = np.random.RandomState(seed)
        
        self.num_samples = num_samples  # number of samples
        self.seq_len = sequence_length  # length of the sequences
        self.vocab = vocab  # list of the vocabulary
        self.vocab_size = len(vocab)  # size of the vocabulary
        self.embedding_dim = embedding_dim  # dimension in the embedding space
        self.n_classes = 2

        # Shuffle the order of the vocabulary
        rs.shuffle(self.vocab)

        

        self.X = []  # List for the design matrix
        n_gen_seqs = 0  # Total Number of generated sequences

        self.y = []  # List for the labels

        # Process for standard positional encoding as Attention is All You Need
        max_pos = context_window
        #position_enc = torch.tensor([[torch.sin(torch.tensor(pos / (10000 ** (i // 2 * 2.0 / self.embedding_dim)), dtype=torch.float)) if i % 2 == 0 else torch.cos(torch.tensor(pos / (10000 ** (i // 2 * 2.0 / self.embedding_dim)), dtype=torch.float)) for i in range(self.embedding_dim)] for pos in range(max_pos)], dtype=torch.float)
        position_enc = torch.tensor([[1 - (0.5 * pos / (max_pos - 1)) for _ in range(embedding_dim)] for pos in range(max_pos - 1, -1, -1)], dtype=torch.float)
        
        stuck_limit = self.seq_len * 5  # Number of iterations that determines if the sequence is cursed, and hence should be dropped
        
        
        # Loop to build the num_samples sequences

        pbar = tqdm(total=self.num_samples, desc='Generating dataset', unit='sample')  # Initialize progress bar

        while(n_gen_seqs < self.num_samples):

            sequence = []  # Initialise the seq to an empty list
            length = 0  # Variable for the length of the sequence
            stuck = 0  # Counter that establish if the algorithm in stuck
            
            while(length < self.seq_len):  # While loop that generates the sequence
                
                
                # If it is the first token, then sample it uniformily from the vocabulary
                if length == 0:
                    
                    while True:
                        # Sample from the distribution
                        #number = rs.zipf(distr_param)
                        number = rs.randint(0, self.vocab_size - 1)
                        # Check if the number is within the range
                        if number < self.vocab_size:
                            break  # Exit the loop if the number is within the range
                    
                    sequence.append(self.vocab[number - 1])
                    length += 1
                    

                else:  #if it is another token then choose it from similarity
                    
                    # Combine token and positional embeddings 
                    combined_embedding = torch.zeros_like(embedding_matrix[0])
                    j = context_window-1
                    for i in range(length - 1, max(length - context_window - 1, -1), -1):
                        token_index = self.vocab.index(sequence[i])
                        combined_embedding += embedding_matrix[token_index] * position_enc[j]
                        j -= 1      
                        
                    #combined_embedding = combined_embedding + position_enc[length - 1]

                    # Calculate similarity with previous tokens and select the most similar one
                    similarities = [torch.dot(embedding_matrix[k], combined_embedding) for k in range(self.vocab_size)]
                    similarities = torch.tensor([similarities]) / temperature
                    probs = nn.functional.softmax(similarities[0], dim=0)
                    probs = probs.numpy()

                    # for k in range(length):
                    #     probs[vocab.index(sequence[k])] = 0

                    # probs /= probs.sum()
                   
                    next_token = rs.choice(vocab, p=probs)
                    
                    sequence.append(next_token)
                    length += 1
                            
                    
                                  
                stuck += 1
                
                # I assumed that if took more then stuck limit iterations to build the sequence, then the seq is cursed
                if(stuck == stuck_limit):
                    stuck = 0
                    sequence = []
                    length = 0
                    
                    
            # Check if the built sequence is already in the dataset
            if(n_gen_seqs != 0):
                is_in_matrix = sequence in self.X
            else:
                is_in_matrix = False  # If it is the first sequence add it in X
            
            
            # If the generated sequence is not already present, build the padded seqs 
            if(not is_in_matrix):
                
                self.X.append(sequence)
                n_gen_seqs += 1
                pbar.update(1)
      
        pbar.close()
        print("\n")
                    
        # Build the target sequences
        for i in range(self.num_samples):
            
            labels = []
            
            label = self.X[i][0] + self.X[i][1]
            labels.append(label)
            
            for j in range(1, self.seq_len - 1):
                #label = self.X[i][j-1] + self.X[i][j] + self.X[i][j+1]
                label = self.X[i][j] + self.X[i][j+1]
                labels.append(label)

            #label = self.X[i][-2] + self.X[i][-1] + self.X[i][0]
            label = self.X[i][-1] + self.X[i][0]
            labels.append(label)
            
            self.y.append(labels) 
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        bound = 2 * np.mean(self.X)# + 2.5 * np.std(self.X)
        self.y = np.where(self.y >= bound, 1, 0)

        

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx],dtype=torch.long), torch.tensor(self.y[idx],dtype=torch.long)
    
    

def read_embeddings(embedding_path, vocab_size):
    embedding_vectors = []
    with open(embedding_path, 'r', encoding = 'utf-8') as f:
        next(f)
        
        lines = f.readlines()
        num_rows = len(lines)

        step = num_rows // vocab_size
        step=1
        for i, line in enumerate(lines):
            if i >= vocab_size * step:
                break
            if i % step == 0:
                values = line.split()
                vector = torch.tensor([float(val) for val in values[1:]])
                embedding_vectors.append(vector)
                    
        embedding_matrix = embedding_vectors
    return embedding_matrix


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

def compute_similarity_matrix(embeddings):
    embeddings = np.array(embeddings)
    similarity = np.dot(embeddings, embeddings.T)
    min_val = np.min(similarity)
    max_val = np.max(similarity)
    
    return (similarity - min_val) / (max_val - min_val)



def match_moments(A, n, axis=0, max_iterations=int(1e6), tolerance=1e-6):
    A = np.array(A)
    B = np.random.randn(*A.shape)
    
    # Get the size of the dimension we're iterating over
    if axis == 0:
        num_slices = A.shape[1]  # Columns
    elif axis == 1:
        num_slices = A.shape[0]  # Rows
    else:
        raise ValueError("Axis must be 0 (columns) or 1 (rows).")
    
    for idx in range(num_slices):
        # Select the appropriate slice (either a row or a column)
        if axis == 0:  # Column-wise
            A_slice = A[:, idx]
            B_slice = B[:, idx]
        else:  # Row-wise
            A_slice = A[idx, :]
            B_slice = B[idx, :]
        
        # Calculate moments of the slice of A
        mean_A = np.mean(A_slice)
        var_A = np.var(A_slice)
        skew_A = moment(A_slice, moment=3) if n >= 3 else None # / np.power(var_A, 1.5) if n >= 3 else None
        kurt_A = moment(A_slice, moment=4) if n >= 4 else None # / (var_A ** 2) if n >= 4 else None
        
        # Match first moment (mean) for slice of B
        B_slice = B_slice - np.mean(B_slice) + mean_A
        
        # Match second moment (variance) for slice of B
        if n >= 2:
            B_slice = (B_slice - np.mean(B_slice))  # Center B_slice
            B_slice = B_slice / np.std(B_slice) * np.sqrt(var_A)  # Adjust variance
            B_slice = B_slice + mean_A  # Adjust mean again
        
        # Iterative process for matching higher moments
        if n > 2:
            for _ in range(int(max_iterations)):
                prev_B_slice = B_slice.copy()
                
                # Compute moments for B_slice
                skew_B = moment(B_slice, moment=3) if n >= 3 else None # / np.power(var_A, 1.5) if n >= 3 else None
                kurt_B = moment(B_slice, moment=4) if n >= 4 else None # / (var_A ** 2) if n >= 4 else None
                
                # Adjust higher moments iteratively
                if n >= 3 and skew_A is not None and np.abs(skew_A - skew_B) > tolerance:
                    B_slice = B_slice * (skew_A / skew_B)
                if n >= 4 and kurt_A is not None and np.abs(kurt_A - kurt_B) > tolerance:
                    B_slice = B_slice * np.sqrt(kurt_A / kurt_B)
                
                # Adjust variance and mean again after adjusting higher moments
                B_slice = B_slice - np.mean(B_slice)  # Center B_slice again
                B_slice = B_slice / np.std(B_slice) * np.sqrt(var_A)  # Adjust variance again
                B_slice = B_slice + mean_A  # Adjust mean again
                
                # Check for convergence
                if np.linalg.norm(B_slice - prev_B_slice) < tolerance:
                    break
        
        # Place the adjusted slice back into B
        if axis == 0:  # Column-wise
            B[:, idx] = B_slice
        else:  # Row-wise
            B[idx, :] = B_slice
    
    return B

##########################################################################################

# Set plot parameters
plt.style.use('science')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14 
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['figure.figsize'] = [15, 8]

##########################################################################################

save_dir = "Empirics/zipf/"
os.makedirs(save_dir, exist_ok=True)

# Define the name of the embeddings
emb_names = ['GloVe', 'Random', #'Preserve Mean',
             'Preserve Two Moments', 'Preserve Three Moments',
             'Preserve Four Moments', 'Preserve Covariance',
             'Preserve Singular Values Distribution']


# Decide to preserve row or column statistics
axis = 0

seed = 42
np.random.seed(seed)

# Generate the embeddings
all_embeddings = []
vocab_size = 5995
labels = [f'{i+1}' for i in range(vocab_size)]
embedding_dim = 50

# Glove
embedding_path = f'Datasets/glove/glove.6B.{embedding_dim}d.txt'
embeddings = read_embeddings(embedding_path, vocab_size)
all_embeddings.append(embeddings)
del embeddings

# Random
embeddings = np.random.random((vocab_size, embedding_dim))
embeddings = [torch.from_numpy(row) for row in embeddings]
all_embeddings.append(embeddings)
del embeddings

# Preserve Row Mean
# embeddings = match_moments(all_embeddings[0], 1)
# embeddings = [torch.from_numpy(row) for row in embeddings]
# all_embeddings.append(embeddings)
# del embeddings

# Preserve Mean & Variance
embeddings = match_moments(all_embeddings[0], 2)
embeddings = [torch.from_numpy(row) for row in embeddings]
all_embeddings.append(embeddings)
del embeddings

# Preserve first three moments
embeddings = match_moments(all_embeddings[0], 3)
embeddings = [torch.from_numpy(row) for row in embeddings]
all_embeddings.append(embeddings)
del embeddings

# Preserve first four moments
embeddings = match_moments(all_embeddings[0], 4)
embeddings = [torch.from_numpy(row) for row in embeddings]
all_embeddings.append(embeddings)
del embeddings

# Preserve Covariance
cov_to_preserve = np.cov(all_embeddings[0], rowvar=False)
epsilon = 1e-8
cov_to_preserve += epsilon * np.eye(cov_to_preserve.shape[0])
L = np.linalg.cholesky(cov_to_preserve)
Z = np.random.normal(0, 1, np.array(all_embeddings[0]).shape)
embeddings = np.dot(Z, L.T)
embeddings = [torch.from_numpy(row) for row in embeddings]
all_embeddings.append(embeddings)
del embeddings

# Preserve Singular Values Distribution
if axis == 0:
    # Preserve Singular Values distribution of rows
    U, singular_values, Vt = np.linalg.svd(all_embeddings[0], full_matrices=False)
    random_U = ortho_group.rvs(dim=U.shape[0])[:, :U.shape[1]]  # Random matrix for rows
    random_V = ortho_group.rvs(dim=Vt.shape[1])[:, :Vt.shape[0]]  # Random matrix for columns
    Sigma = np.diag(singular_values)
    embeddings = random_U @ Sigma @ random_V.T  # Reconstruct matrix
elif axis == 1:
    # Preserve Singular Values distribution of columns
    V, singular_values, Ut = np.linalg.svd(np.array(all_embeddings[0]).T.tolist(), full_matrices=False)
    random_V = ortho_group.rvs(dim=V.shape[0])[:, :V.shape[1]]  # Random matrix for columns
    random_U = ortho_group.rvs(dim=Ut.shape[1])[:, :Ut.shape[0]]  # Random matrix for rows
    Sigma = np.diag(singular_values)
    embeddings = random_V @ Sigma @ random_U.T  # Reconstruct matrix
    embeddings = embeddings.T

embeddings = [torch.from_numpy(row) for row in embeddings]
all_embeddings.append(embeddings)
del embeddings


all_similarity = []
all_singular_values = []
all_covariance = []
all_variance = []
all_third_moment = []
all_fourth_moment = []

# Plot heatmap of the GloVe embeddings
plt.rcParams['xtick.major.size'] = 5  # Major x-tick size
plt.rcParams['ytick.major.size'] = 5  # Major y-tick size
plt.rcParams['xtick.minor.size'] = 0  # Minor x-tick size
plt.rcParams['ytick.minor.size'] = 0  # Minor y-tick size

plt.figure()
plt.title(f'Embeddings for {emb_names[0]}')
sns.heatmap(all_embeddings[0], cmap='viridis', cbar_kws={'label': 'Magnitude'})
plt.xlabel('Dimension')
plt.ylabel('Token')
plt.savefig(os.path.join(save_dir, 'GloVe_embs.png'), format='png', dpi=200)
plt.clf()
plt.close()

# Similarity matrix between tokens
sim_mtrx = compute_similarity_matrix(all_embeddings[0])
all_similarity.append(sim_mtrx)

# Compute and plot covariance
cov = np.cov(all_embeddings[0], rowvar=False)
all_covariance.append(cov)
plt.figure()
plt.title(f'Covariance for {emb_names[0]}')
sns.heatmap(cov, cmap='viridis', cbar_kws={'label': 'Covariance'})
plt.xlabel('Token')
plt.ylabel('Token')
plt.savefig(os.path.join(save_dir, 'GloVe_cov.png'), format='png', dpi=200)
plt.clf()
plt.close()

plt.rcdefaults()
plt.style.use('science')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.titlesize'] = 16 
plt.rcParams['figure.figsize'] = [15, 8]

# Singular values
_, singular_values, _ = np.linalg.svd(all_embeddings[0], full_matrices=False)
all_singular_values.append(singular_values)

variance = np.var(all_embeddings[0], axis=axis)
all_variance.append(variance)

moments = []
embs = np.array(all_embeddings[0])
for j in range(embs.shape[1]):
    mean_emb = np.mean(embs[:, j])
    moment_emb = np.mean((embs[:, j] - mean_emb) ** 3)
    moments.append(moment_emb)

all_third_moment.append(moments)

moments = []
for j in range(embs.shape[1]):
    mean_emb = np.mean(embs[:, j])
    moment_emb = np.mean((embs[:, j] - mean_emb) ** 4)
    moments.append(moment_emb)

all_fourth_moment.append(moments)

# Precompute the norms of the GloVe statistics
glove_embedding_norm = np.linalg.norm(all_embeddings[0])
glove_covariance_norm = np.linalg.norm(all_covariance[0])

for idx, emb_name in enumerate(emb_names[1:], start=1):

    sim_mtrx = compute_similarity_matrix(all_embeddings[0])
    all_similarity.append(sim_mtrx)

    cov = np.cov(all_embeddings[idx], rowvar=False)
    all_covariance.append(cov)

    _, singular_values, _ = np.linalg.svd(all_embeddings[idx], full_matrices=False)
    all_singular_values.append(singular_values)

    variance = np.var(all_embeddings[idx], axis=axis)
    all_variance.append(variance)


    moments = []
    embs = np.array(all_embeddings[idx])
    for j in range(embs.shape[1]):
        mean_emb = np.mean(embs[:, j])
        moment_emb = np.mean((embs[:, j] - mean_emb) ** 3)
        moments.append(moment_emb)

    all_third_moment.append(moments)

    moments = []
    for j in range(embs.shape[1]):
        mean_emb = np.mean(embs[:, j])
        moment_emb = np.mean((embs[:, j] - mean_emb) ** 4)
        moments.append(moment_emb)

    all_fourth_moment.append(moments)

    
    # Compute distances for each metric with respect to GloVe
    dist_embedding = np.abs(glove_embedding_norm - np.linalg.norm(all_embeddings[idx]))
    dist_singular = np.linalg.norm(all_singular_values[0] - all_singular_values[idx])
    dist_covariance = np.abs(glove_covariance_norm - np.linalg.norm(all_covariance[idx]))
    dist_variance = np.linalg.norm(np.array(all_variance[0]) - np.array(all_variance[idx]))
    dist_third = np.linalg.norm(np.array(all_third_moment[0]) - np.array(all_third_moment[idx]))
    dist_fourth = np.linalg.norm(np.array(all_fourth_moment[0]) - np.array(all_fourth_moment[idx]))


    print(f"--- Distances between GloVe and {emb_name} ---")
    print(f"Embedding Distance: {dist_embedding:.6f}")
    print(f"Singular Value Distance: {dist_singular:.6f}")
    print(f"Covariance Distance: {dist_covariance:.6f}")
    # print(f"Mean Distance: {dist_mean:.6f}")
    print(f"Variance Distance: {dist_variance:.6f}")
    print(f"Third Moment Distance: {dist_third:.6f}")
    print(f"Fourth Moment Distance: {dist_fourth:.6f}")
    print("\n")

plt.rcParams['xtick.major.size'] = 5 
plt.rcParams['ytick.major.size'] = 5  
plt.rcParams['xtick.minor.size'] = 0  
plt.rcParams['ytick.minor.size'] = 0    
plt.rcParams['axes.labelsize'] = 10 
plt.rcParams['axes.titlesize'] = 12 
plt.rcParams['xtick.labelsize'] = 8 
plt.rcParams['ytick.labelsize'] = 8

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()

for idx in range(1, min(len(all_embeddings), 7)):

    sns.heatmap(all_embeddings[idx], ax=axs[idx - 1], cmap='viridis', cbar_kws={'label': 'Magnitude'})
    axs[idx - 1].set_title(f'{emb_names[idx]}')
    axs[idx - 1].set_xlabel('Dimension')
    axs[idx - 1].set_ylabel('Token')

for i in range(len(all_embeddings), 6):
    axs[i].set_visible(False)

fig.suptitle('Embeddings', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'embedding_comparison.png'), format='png', dpi=200)
plt.clf()
plt.close()

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()

for idx in range(1, len(all_covariance)):
    # Plot heatmap for covariance
    sns.heatmap(all_covariance[idx], ax=axs[idx - 1], cmap='viridis', cbar_kws={'label': 'Covariance'})
    axs[idx - 1].set_title(f'{emb_names[idx]}')
    axs[idx - 1].set_xlabel('Token')
    axs[idx - 1].set_ylabel('Token')

for i in range(len(all_covariance), 6):
    axs[i].set_visible(False)

fig.suptitle('Covariance', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'covariance_comparison.png'), format='png', dpi=200)
plt.clf()
plt.close()

plt.rcdefaults()
plt.style.use('science')
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10 
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['figure.figsize'] = [15, 8]


# Plot Singular Values
base_line_width = 1
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()
colors = ListedColormap([
    '#E41A1C',  # Red
    '#377EB8',  # Blue
    '#4DAF4A',  # Green
    '#FF7F00',  # Orange
    '#984EA3',  # Purple
    '#B3B6E0',   # Light Purple
    '#66C2A5'   # Light Blue
])



for idx in range(1, len(all_singular_values)):

    axs[idx - 1].plot(np.arange(1, len(all_singular_values[0]) + 1), all_singular_values[0],
                      color=colors(0), linewidth=base_line_width, marker='o', label=f'{emb_names[0]}')

    
    axs[idx - 1].plot(np.arange(1, len(all_singular_values[idx]) + 1), all_singular_values[idx],
                      color=colors(idx / len(all_singular_values)), marker='p',
                      linewidth=base_line_width, label=f'{emb_names[idx]}')

    axs[idx - 1].set_title(f'{emb_names[0]} VS {emb_names[idx]}')
    axs[idx - 1].set_xlabel('Index')
    #axs[idx - 1].set_yscale('log')
    axs[idx - 1].set_ylabel('Singular Value')
    axs[idx - 1].legend()


for i in range(len(all_singular_values), 6):
    axs[i].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'SV_comparison.png'), format='png', dpi=200)
plt.clf()
plt.close()

# Plot Variance
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()

for idx in range(1, len(all_variance)):

    axs[idx - 1].plot(np.arange(1, len(all_variance[0]) + 1), all_variance[0],
                      color=colors(0), linewidth=base_line_width, marker='o', label=f'{emb_names[0]}')

    axs[idx - 1].plot(np.arange(1, len(all_variance[idx]) + 1), all_variance[idx],
                      color=colors(idx / len(all_variance)), linewidth=base_line_width, marker='p', label=f'{emb_names[idx]}')


    axs[idx - 1].set_title(f'{emb_names[0]} VS {emb_names[idx]}')
    axs[idx - 1].set_xlabel('Index')
    axs[idx - 1].set_ylabel('Variance')
    axs[idx - 1].legend()

for i in range(len(all_variance), 6):
    axs[i].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'variance_comparison.png'), format='png', dpi=200)
plt.clf()
plt.close()


# Plot Third Moment
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()

for idx in range(1, len(all_third_moment)):

    axs[idx - 1].plot(np.arange(1, len(all_third_moment[0]) + 1), all_third_moment[0],
                      color=colors(0), linewidth=base_line_width, marker='o', label=f'{emb_names[0]}')

    axs[idx - 1].plot(np.arange(1, len(all_third_moment[idx]) + 1), all_third_moment[idx],
                      color=colors(idx / len(all_third_moment)), linewidth=base_line_width, marker='p', label=f'{emb_names[idx]}')

    axs[idx - 1].set_title(f'{emb_names[0]} VS {emb_names[idx]}')
    axs[idx - 1].set_xlabel('Index')
    axs[idx - 1].set_ylabel('Third Moment')
    axs[idx - 1].legend()

for i in range(len(all_third_moment), 6):
    axs[i].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'third_moment_comparison.png'), format='png', dpi=200)
plt.clf()
plt.close()

# Plot Fourth Moment
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()

for idx in range(1, len(all_fourth_moment)):

    axs[idx - 1].plot(np.arange(1, len(all_fourth_moment[0]) + 1), all_fourth_moment[0],
                      color=colors(0), linewidth=base_line_width, marker='o', label=f'{emb_names[0]}')

    axs[idx - 1].plot(np.arange(1, len(all_fourth_moment[idx]) + 1), all_fourth_moment[idx],
                      color=colors(idx / len(all_fourth_moment)), linewidth=base_line_width, marker='p', label=f'{emb_names[idx]}')

    axs[idx - 1].set_title(f'{emb_names[0]} VS {emb_names[idx]}')
    axs[idx - 1].set_xlabel('Index')
    axs[idx - 1].set_ylabel('Fourth Moment')
    axs[idx - 1].legend()

for i in range(len(all_fourth_moment), 6):
    axs[i].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'fourth_moment_comparison.png'), format='png', dpi=200)
plt.clf()
plt.close()


# Hyperparameters
num_samples = 50000
sequence_length = 10
context_window = 10
# vocab_size = 5995
vocab = list(range(vocab_size))
distr_param = 1.1
temperature = 2


filename = f"all_sequences_num_samples{num_samples}_seq_len{sequence_length}_context_win{context_window}_vocab_size{vocab_size}_distr_param{distr_param}_temperature{temperature}.pkl"
print(f"{filename}\n")

if os.path.exists(filename):
    with open(filename, 'rb') as f:
        all_sequences = pickle.load(f)
    print(f"Loaded existing sequences from {filename}.")
else:
    all_sequences = []
    
    for idx in range(len(emb_names)):
        dataset = PrimitiveNLP(num_samples, sequence_length, context_window, vocab, all_embeddings[idx], seed, distr_param)
        all_sequences.append(dataset.X)

    with open(filename, 'wb') as f:
        pickle.dump(all_sequences, f)
    print(f"Generated and saved sequences to {filename}.")


plt.rcdefaults()
plt.style.use('science')
plt.rcParams['axes.labelsize'] = 10  
plt.rcParams['legend.fontsize'] = 10 
plt.rcParams['axes.titlesize'] = 12 
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['figure.figsize'] = [15, 8]
colors = plt.cm.Set2


line_styles = {
    0: '-',     # Solid line for idx = 0
    1: '--',    # Dashed line for idx = 1
    2: ':',    # Dash-dot line for idx = 2 to 4
    3: ':',    # Dash-dot line for idx = 2 to 4
    4: ':',    # Dash-dot line for idx = 2 to 4
    5: ':',     # Dotted line for idx = 5
    6: (0, (5, 5, 1, 5)),  # Custom dash pattern for idx = 6
}


for idx, sequences in enumerate(all_sequences):
    distr = []

    unique_tokens, counts = np.unique(sequences, return_counts=True)

    num_tokens = num_samples * sequence_length
    distr = np.array(counts) / num_tokens

    distr[::-1].sort()

    diff = vocab_size - len(distr)
    if diff != 0 :
        distr = np.append(distr, np.zeros(diff))

    style = line_styles.get(idx, '-')

    plt.plot(range(1, len(distr) + 1), distr, color=colors(idx / len(all_sequences)), 
             linestyle=style, label=f'{emb_names[idx]}', linewidth=2)
    

freqs = np.loadtxt('Datasets/frequencies.txt')
freqs = freqs / np.sum(freqs)
freqs = np.sort(freqs)[::-1]

plt.plot(range(1, len(distr) + 1), freqs, '-*', markersize=4, label='Tiny Stories', color='darkred')

plt.xlabel('Rank', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=14)
plt.title('Token Frequency Distribution', fontsize=16)
plt.savefig(os.path.join(save_dir, 'embs_comparison.png'), format='png', dpi=200)
plt.clf()
plt.close()