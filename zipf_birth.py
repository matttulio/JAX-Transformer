import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from torch.utils.data import Dataset
import jax.numpy as jnp
from scipy.stats import skew, kurtosis, moment, ortho_group
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
        next(f)  # Skip the header or first line if any
        # Use the readlines() method to read all lines into a list
        lines = f.readlines()

        # Count the number of lines in the list
        num_rows = len(lines)

        step = num_rows // vocab_size
        step=1
        for i, line in enumerate(lines):
            if i >= vocab_size * step:  # Break if enough vectors are read
                break
            if i % step == 0:  # Only take every step-th vector
                values = line.split()
                vector = torch.tensor([float(val) for val in values[1:]])
                embedding_vectors.append(vector)
                    
        embedding_matrix = embedding_vectors
    return embedding_matrix


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def compute_similarity_matrix(normalized_embeddings):
    return np.dot(normalized_embeddings, normalized_embeddings.T)



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
        skew_A = moment(A_slice, moment=3) / np.power(var_A, 1.5) if n >= 3 else None
        kurt_A = moment(A_slice, moment=4) / (var_A ** 2) if n >= 4 else None
        
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
                mean_B = np.mean(B_slice)
                var_B = np.var(B_slice)
                skew_B = moment(B_slice, moment=3) / np.power(var_B, 1.5) if n >= 3 else None
                kurt_B = moment(B_slice, moment=4) / (var_B ** 2) if n >= 4 else None
                
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

plt.style.use('science')
plt.rcParams['figure.figsize'] = [15, 8]

# Define the name of the embeddings
emb_names = ['GloVe', 'Random', #'Preserve Mean',
             'Preserve Two Moments', 'Preserve Three Moments',
             'Preserve Four Moments', 'Preserve Covariance',
             'Preserve Eigenvalue Distribution']


# Decide to preserve row or column statistics
axis = 0


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
embeddings = match_moments(all_embeddings[0], 2, axis)
embeddings = [torch.from_numpy(row) for row in embeddings]
all_embeddings.append(embeddings)
del embeddings

# Preserve first three moments
embeddings = match_moments(all_embeddings[0], 3, axis)
embeddings = [torch.from_numpy(row) for row in embeddings]
all_embeddings.append(embeddings)
del embeddings

# Preserve first four moments
embeddings = match_moments(all_embeddings[0], 4, axis)
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

# Preserve Eigenvalue Distribution
if axis == 0:
    # Preserve eigenvalue distribution of rows
    U, singular_values, Vt = np.linalg.svd(all_embeddings[0], full_matrices=False)
    random_U = ortho_group.rvs(dim=U.shape[0])[:, :U.shape[1]]  # Random matrix for rows
    random_V = ortho_group.rvs(dim=Vt.shape[1])[:, :Vt.shape[0]]  # Random matrix for columns
    Sigma = np.diag(singular_values)
    embeddings = random_U @ Sigma @ random_V.T  # Reconstruct matrix
elif axis == 1:
    # Preserve eigenvalue distribution of columns
    V, singular_values, Ut = np.linalg.svd(np.array(all_embeddings[0]).T.tolist(), full_matrices=False)
    random_V = ortho_group.rvs(dim=V.shape[0])[:, :V.shape[1]]  # Random matrix for columns
    random_U = ortho_group.rvs(dim=Ut.shape[1])[:, :Ut.shape[0]]  # Random matrix for rows
    Sigma = np.diag(singular_values)
    embeddings = random_V @ Sigma @ random_U.T  # Reconstruct matrix
    embeddings = embeddings.T

embeddings = [torch.from_numpy(row) for row in embeddings]
all_embeddings.append(embeddings)
del embeddings

# Initialize lists to store values
all_similarity = []
all_singular_values = []
all_covariance = []
# all_mean = []
all_variance = []
all_skewness = []
all_kurtosis = []

# # Plot heatmap of the GloVe embeddings
# plt.figure()
# plt.title(f'Embeddings for {emb_names[0]}')
# sns.heatmap(all_embeddings[0], yticklabels=False, cmap='viridis', cbar_kws={'label': 'Magnitude'})
# plt.xlabel('Dimension')
# plt.ylabel('Token')
# plt.xticks(fontsize=7)
# plt.yticks(fontsize=7)
# plt.clf()
# plt.close()

# # Plot similarity matrix between tokens
# norm_embs = normalize_embeddings(all_embeddings[0])
# sim_mtrx = compute_similarity_matrix(norm_embs)
# all_similarity.append(sim_mtrx)
# plt.figure()
# plt.title(f'Normalized similarity matrix for {emb_names[0]}')
# sns.heatmap(sim_mtrx, yticklabels=False, cmap='viridis', cbar_kws={'label': 'Similarity'})
# plt.xlabel('Token')
# plt.ylabel('Token')
# plt.xticks(fontsize=7)
# plt.yticks(fontsize=7)
# plt.clf()
# plt.close()

# # Compute and plot singular values
# U, singular_values, Vt = np.linalg.svd(all_embeddings[0], full_matrices=False)
# all_singular_values.append(singular_values)
# plt.scatter(np.arange(1, len(singular_values) + 1), singular_values, color='b')
# plt.title(f'Singular Values for {emb_names[0]}')
# plt.xlabel('Index')
# plt.ylabel('Singular Value')
# plt.grid(True)
# plt.clf()
# plt.close()

# # Compute and plot covariance
# cov = np.cov(all_embeddings[0], rowvar=False)
# all_covariance.append(cov)
# plt.figure()
# plt.title(f'Covariance for {emb_names[0]}')
# sns.heatmap(cov, cmap='viridis', cbar_kws={'label': 'Covariance'})
# plt.xlabel('Token')
# plt.ylabel('Token')
# plt.xticks(fontsize=7)
# plt.yticks(fontsize=7)
# plt.clf()
# plt.close()

# # Compute statistical properties (variance, skewness, kurtosis)
# # mean = np.mean(all_embeddings[0], axis=0)
# # all_mean.append(mean)
# # plt.scatter(np.arange(1, embedding_dim + 1), mean, color='b')
# # plt.title(f'Mean for {emb_names[0]}')
# # plt.xlabel('Token')
# # plt.ylabel('Mean')
# # plt.grid(True)
# # plt.clf()
# # plt.close()

# variance = np.var(all_embeddings[0], axis=axis)
# all_variance.append(variance)
# if axis == 0:
#     plt.scatter(np.arange(1, embedding_dim + 1), variance, color='b')
# elif axis == 1:
#     plt.scatter(np.arange(1, vocab_size + 1), variance, color='b')
# plt.title(f'Variance for {emb_names[0]}')
# plt.xlabel('Token')
# plt.ylabel('Variance')
# plt.grid(True)
# plt.clf()
# plt.close()

# skewness = skew(all_embeddings[0], axis=axis)
# all_skewness.append(skewness)
# if axis == 0:
#     plt.scatter(np.arange(1, embedding_dim + 1), skewness, color='b')
# elif axis == 1:
#     plt.scatter(np.arange(1, vocab_size + 1), skewness, color='b')
# plt.title(f'Skewness for {emb_names[0]}')
# plt.xlabel('Token')
# plt.ylabel('Skewness')
# plt.grid(True)
# plt.clf()
# plt.close()

# kurt = kurtosis(all_embeddings[0], axis=axis)
# all_kurtosis.append(kurt)
# if axis == 0:
#     plt.scatter(np.arange(1, embedding_dim + 1), kurt, color='b')
# elif axis == 1:
#     plt.scatter(np.arange(1, vocab_size + 1), kurt, color='b')
# plt.title(f'Kurtosis for {emb_names[0]}')
# plt.xlabel('Token')
# plt.ylabel('Kurtosis')
# plt.grid(True)
# plt.clf()
# plt.close()

# # Precompute the norms of the GloVe (first embedding) statistics
# glove_embedding_norm = np.linalg.norm(all_embeddings[0])
# glove_singular_norm = np.linalg.norm(all_singular_values[0])
# glove_covariance_norm = np.linalg.norm(all_covariance[0])
# #glove_mean_norm = np.linalg.norm(all_mean[0])
# glove_variance_norm = np.linalg.norm(all_variance[0])
# glove_skewness_norm = np.linalg.norm(all_skewness[0])
# glove_kurtosis_norm = np.linalg.norm(all_kurtosis[0])

# for idx, emb_name in enumerate(emb_names[1:], start=1):
 
#     # Plot heatmap of the GloVe embeddings
#     plt.figure()
#     plt.title(f'Embeddings for {emb_name}')
#     sns.heatmap(all_embeddings[idx], yticklabels=False, cmap='viridis', cbar_kws={'label': 'Magnitude'})
#     plt.xlabel('Dimension')
#     plt.ylabel('Token')
#     plt.xticks(fontsize=7)
#     plt.yticks(fontsize=7)
#     plt.clf()
#     plt.close()

#     # Plot similarity matrix between tokens
#     norm_embs = normalize_embeddings(all_embeddings[idx])
#     sim_mtrx = compute_similarity_matrix(norm_embs)
#     all_similarity.append(sim_mtrx)
#     plt.figure()
#     plt.title(f'Normalized similarity matrix for {emb_name}')
#     sns.heatmap(sim_mtrx, yticklabels=False, cmap='viridis', cbar_kws={'label': 'Similarity'})
#     plt.xlabel('Token')
#     plt.ylabel('Token')
#     plt.xticks(fontsize=7)
#     plt.yticks(fontsize=7)
#     plt.clf()
#     plt.close()

#     # Compute and plot singular values
#     U, singular_values, Vt = np.linalg.svd(all_embeddings[idx], full_matrices=False)
#     all_singular_values.append(singular_values)
#     plt.scatter(np.arange(1, len(singular_values) + 1), singular_values, color='b')
#     plt.title(f'Singular Values for {emb_name}')
#     plt.xlabel('Index')
#     plt.ylabel('Singular Value')
#     plt.grid(True)
#     plt.clf()
#     plt.close()

#     # Compute and plot covariance
#     cov = np.cov(all_embeddings[idx], rowvar=False)
#     all_covariance.append(cov)
#     plt.figure()
#     plt.title(f'Covariance for {emb_name}')
#     sns.heatmap(cov, cmap='viridis', cbar_kws={'label': 'Covariance'})
#     plt.xlabel('Token')
#     plt.ylabel('Token')
#     plt.xticks(fontsize=7)
#     plt.yticks(fontsize=7)
#     plt.clf()
#     plt.close()

#     # Compute statistical properties (variance, skewness, kurtosis)
#     # mean = np.mean(all_embeddings[idx], axis=0)
#     # all_mean.append(mean)
#     # plt.scatter(np.arange(1, embedding_dim + 1), mean, color='b')
#     # plt.title(f'Mean for {emb_name}')
#     # plt.xlabel('Token')
#     # plt.ylabel('Mean')
#     # plt.grid(True)
#     # plt.clf()
#     # plt.close()

#     variance = np.var(all_embeddings[idx], axis=axis)
#     all_variance.append(variance)
#     if axis == 0:
#         plt.scatter(np.arange(1, embedding_dim + 1), variance, color='b')
#     elif axis == 1:
#         plt.scatter(np.arange(1, vocab_size + 1), variance, color='b')
#     plt.title(f'Variance for {emb_name}')
#     plt.xlabel('Token')
#     plt.ylabel('Variance')
#     plt.grid(True)
#     plt.clf()
#     plt.close()

#     skewness = skew(all_embeddings[idx], axis=axis)
#     all_skewness.append(skewness)
#     if axis == 0:
#         plt.scatter(np.arange(1, embedding_dim + 1), skewness, color='b')
#     elif axis == 1:
#         plt.scatter(np.arange(1, vocab_size + 1), skewness, color='b')
#     plt.title(f'Skewness for {emb_name}')
#     plt.xlabel('Token')
#     plt.ylabel('Skewness')
#     plt.grid(True)
#     plt.clf()
#     plt.close()

#     kurt = kurtosis(all_embeddings[idx], axis=axis)
#     all_kurtosis.append(kurt)
#     if axis == 0:
#         plt.scatter(np.arange(1, embedding_dim + 1), kurt, color='b')
#     elif axis == 1:
#         plt.scatter(np.arange(1, vocab_size + 1), kurt, color='b')
#     plt.title(f'Kurtosis for {emb_name}')
#     plt.xlabel('Token')
#     plt.ylabel('Kurtosis')
#     plt.grid(True)
#     plt.clf()
#     plt.close()

    
#     # Compute distances for each metric with respect to GloVe
#     dist_embedding = np.abs(glove_embedding_norm - np.linalg.norm(all_embeddings[idx]))
#     dist_singular = np.abs(glove_singular_norm - np.linalg.norm(all_singular_values[idx]))
#     dist_covariance = np.abs(glove_covariance_norm - np.linalg.norm(all_covariance[idx]))
#     # dist_mean = np.abs(glove_mean_norm - np.linalg.norm(all_mean[idx]))
#     dist_variance = np.abs(glove_variance_norm - np.linalg.norm(all_variance[idx]))
#     dist_skewness = np.abs(glove_skewness_norm - np.linalg.norm(all_skewness[idx]))
#     dist_kurtosis = np.abs(glove_kurtosis_norm - np.linalg.norm(all_kurtosis[idx]))


#     # Print the results with formatted output
#     print(f"--- Distances between GloVe and {emb_name} ---")
#     print(f"Embedding Distance: {dist_embedding:.4f}")
#     print(f"Singular Value Distance: {dist_singular:.4f}")
#     print(f"Covariance Distance: {dist_covariance:.4f}")
#     # print(f"Mean Distance: {dist_mean:.4f}")
#     print(f"Variance Distance: {dist_variance:.4f}")
#     print(f"Skewness Distance: {dist_skewness:.4f}")
#     print(f"Kurtosis Distance: {dist_kurtosis:.4f}")
#     print("\n")





# base_line_width = 3.5  # Base line width for the largest line

# # Plot Singular Values
colors = plt.colormaps.get_cmap('Set2')
# for idx, singular_values in enumerate(all_singular_values):
#     line_width = base_line_width + (base_line_width * (1 - idx * 1.75 / len(all_singular_values)))  # Decreasing line width
#     plt.plot(np.arange(1, len(singular_values) + 1), singular_values, 
#              color=colors(idx / len(all_singular_values)), 
#              label=f'{emb_names[idx]}', linewidth=line_width)

# plt.title('Scatter Plot of Singular Values')
# plt.xlabel('Index')
# plt.yscale('log')
# plt.ylabel('Singular Value')
# plt.legend()
# # plt.show()

# # # Plot Mean
# # for idx, mean in enumerate(all_mean):
# #     line_width = base_line_width + (base_line_width * (1 - idx * 1.75 / len(all_mean)))
# #     plt.plot(np.arange(1, len(mean) + 1), mean, 
# #              color=colors(idx / len(all_mean)), 
# #              label=f'{emb_names[idx]}', linewidth=line_width)
# 
# # plt.title('Scatter Plot of Mean')
# # plt.xlabel('Index')
# # plt.yscale('symlog')
# # plt.ylabel('Mean')
# # plt.legend()
# # # plt.show()

# # Plot Variance
# for idx, var in enumerate(all_variance):
#     line_width = base_line_width + (base_line_width * (1 - idx * 1.75 / len(all_variance)))
#     plt.plot(np.arange(1, len(var) + 1), var, 
#              color=colors(idx / len(all_variance)), 
#              label=f'{emb_names[idx]}', linewidth=line_width)

# plt.title('Scatter Plot of Variance')
# plt.xlabel('Index')
# plt.yscale('log')
# plt.ylabel('Variance')
# plt.legend()
# # plt.show()
# plt.clf()
# plt.close()

# # Plot Skewness
# for idx, skews in enumerate(all_skewness):
#     line_width = base_line_width + (base_line_width * (1 - idx * 1.75 / len(all_skewness)))
#     plt.plot(np.arange(1, len(skews) + 1), skews, 
#              color=colors(idx / len(all_skewness)), 
#              label=f'{emb_names[idx]}', linewidth=line_width)

# plt.title('Scatter Plot of Skewness')
# plt.xlabel('Index')
# plt.yscale('symlog')
# plt.ylabel('Skewness')
# plt.legend()
# # plt.show()
# plt.clf()
# plt.close()

# # Plot Kurtosis
# for idx, kurts in enumerate(all_kurtosis):
#     line_width = base_line_width + (base_line_width * (1 - idx * 1.75 / len(all_kurtosis)))
#     plt.plot(np.arange(1, len(kurts) + 1), kurts, 
#              color=colors(idx / len(all_kurtosis)), 
#              label=f'{emb_names[idx]}', linewidth=line_width)

# plt.title('Scatter Plot of Kurtosis')
# plt.xlabel('Index')
# plt.yscale('symlog')
# plt.ylabel('Kurtosis')
# plt.legend()
# # plt.show()
# plt.clf()
# plt.close()


# Hyperparameters
num_samples = 50000
sequence_length = 10
context_window = 10
# vocab_size = 5995
vocab = list(range(vocab_size))
seed = 42
distr_param = 1.1
temperature = 3

# Generate a unique filename based on hyperparameters
filename = f"all_sequences_num_samples{num_samples}_seq_len{sequence_length}_context_win{context_window}_vocab_size{vocab_size}_distr_param{distr_param}_temperature{temperature}.pkl"

# Check if the file already exists
if os.path.exists(filename):
    with open(filename, 'rb') as f:
        all_sequences = pickle.load(f)
    print(f"Loaded existing sequences from {filename}.")
else:
    all_sequences = []
    
    # Generate sequences if the file does not exist
    for idx in range(len(emb_names)):
        dataset = PrimitiveNLP(num_samples, sequence_length, context_window, vocab, all_embeddings[idx], seed, distr_param)
        all_sequences.append(dataset.X)

    # Save the sequences to a file
    with open(filename, 'wb') as f:
        pickle.dump(all_sequences, f)
    print(f"Generated and saved sequences to {filename}.")


# Define your different line styles
line_styles = {
    0: '-',     # Solid line for idx = 0
    1: '--',    # Dashed line for idx = 1
    2: ':',    # Dash-dot line for idx = 2 to 4
    3: ':',    # Dash-dot line for idx = 2 to 4
    4: ':',    # Dash-dot line for idx = 2 to 4
    5: ':',     # Dotted line for idx = 5
    6: (0, (5, 5, 1, 5)),  # Custom dash pattern for idx = 6
}

plt.figure(figsize=(12, 8))


# Iterate through all sublists in all_sequences
for idx, sequences in enumerate(all_sequences):
    distr = []

    unique_tokens, counts = np.unique(sequences, return_counts=True)

    # Compute the frequencies of each token
    num_tokens = num_samples * sequence_length  # Total number of tokens in the current batch of sequences
    distr = np.array(counts) / num_tokens  # Normalize by total number of tokens

    # Sort the frequencies in descending order
    distr[::-1].sort()

    diff = vocab_size - len(distr)
    if diff != 0 :
        distr = np.append(distr, np.zeros(diff))


    # Determine the appropriate line style for the current index
    style = line_styles.get(idx, '-')  # Default to solid line if idx not in dictionary
    
    # Plot the distribution with explicit linestyle and color keyword arguments
    plt.plot(range(1, len(distr) + 1), distr, color=colors(idx / len(all_sequences)), 
             linestyle=style, label=f'{emb_names[idx]}', linewidth=2)
    

freqs = np.loadtxt('Datasets/frequencies.txt')
freqs = freqs / np.sum(freqs)
freqs = np.sort(freqs)[::-1]

plt.plot(range(1, len(distr) + 1), freqs, '-*', markersize=4, label='Tiny Stories', color='darkred')

# Plot formatting
plt.xlabel('Degree', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=14)
plt.title('Distribution of Tokens Across Sequences', fontsize=16)
plt.show()