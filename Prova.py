import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

def read_logspaced_embeddings(embedding_path, vocab_size):
    embedding_vectors = []
    with open(embedding_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip the header or first line if any
        lines = f.readlines()
        num_rows = len(lines)

        if vocab_size > num_rows:
            raise ValueError(f"vocab_size ({vocab_size}) cannot be greater than the number of rows in the file ({num_rows})")
        
        for i in range(num_rows):
            indices = np.unique(np.logspace(np.log(i+1), np.log(num_rows), num=vocab_size-1, base=np.e, endpoint=True, dtype=int))
            if(len(indices) == vocab_size-1):
                indices = np.insert(indices, 0, 0)
                break

        if(len(indices) != vocab_size):
            raise ValueError("It was not possible to correctly sample logspace embeddings")


        for idx in indices:
            line = lines[idx]
            values = line.split()
            vector = torch.tensor([float(val) for val in values[1:]])
            embedding_vectors.append(vector)
    return torch.stack(embedding_vectors).numpy()

def read_embeddings(embedding_path, vocab_size):
    embedding_vectors = []
    with open(embedding_path, 'r', encoding = 'utf-8') as f:
        next(f)  # Skip the header or first line if any
        # Use the readlines() method to read all lines into a list
        lines = f.readlines()

        # Count the number of lines in the list
        num_rows = len(lines)

        step = num_rows // vocab_size
        #step=1
        for i, line in enumerate(lines):
            if i >= vocab_size * step:  # Break if enough vectors are read
                break
            if i % step == 0:  # Only take every step-th vector
                values = line.split()
                vector = torch.tensor([float(val) for val in values[1:]])
                embedding_vectors.append(vector)
    return torch.stack(embedding_vectors).numpy()

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def compute_similarity_matrix(normalized_embeddings):
    return np.dot(normalized_embeddings, normalized_embeddings.T)


embedding_path = 'Datasets/glove/glove.6B.50d.txt'
vocab_size = 78
labels = [f'{i+1}' for i in range(vocab_size)]
embedding_dim = 50
embeddings = read_embeddings(embedding_path, vocab_size)
#embeddings = nn.Embedding(vocab_size, embedding_dim)  # embedding layer
#embeddings = (embeddings.weight.data).cpu().detach().numpy()


plt.figure(figsize=(10, 8))
sns.heatmap(embeddings, yticklabels=labels, cmap='viridis')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
#plt.show()


cov_A = np.cov(embeddings, rowvar=False)
epsilon = 1e-8  # A small value to ensure positive definiteness
cov_A += epsilon * np.eye(cov_A.shape[0])
L = np.linalg.cholesky(cov_A)
Z = np.random.normal(0, 1, embeddings.shape)
B = np.dot(Z, L.T)

# plt.figure(figsize=(10, 8))
# sns.heatmap(B, yticklabels=labels, cmap='viridis')
# plt.xticks(fontsize=7)
# plt.yticks(fontsize=7)
#plt.show()

from scipy.stats import ortho_group  # To generate random orthogonal matrices
A = embeddings

# Step 1: SVD of matrix A
U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)

# Step 2: Generate random orthogonal matrices with appropriate dimensions
random_U = ortho_group.rvs(dim=U.shape[0])[:, :U.shape[1]]  # Truncate U to match A
random_V = ortho_group.rvs(dim=Vt.shape[0])

# Step 3: Construct Sigma as a diagonal matrix of singular values
Sigma = np.diag(singular_values)

# Step 4: Reconstruct B with the same singular values as A
B = random_U @ Sigma @ random_V

B = [torch.from_numpy(row) for row in B]



#masked_matrix = similarity_matrix.astype(float)
# np.fill_diagonal(masked_matrix, np.nan)

# Flatten the masked matrix and remove np.nan values
#flattened_masked_matrix = masked_matrix.flatten()
#non_diagonal_elements = flattened_masked_matrix[~np.isnan(flattened_masked_matrix)]

# # Get unique values and their counts
# values, counts = np.unique(similarity_matrix, return_counts=True)

# # Print the unique values and their counts
# print("Unique values (excluding diagonal):", values[np.where(counts>2)])
# print("Counts:", counts[np.where(counts>2)])
# print("Number of unique values (excluding diagonal):", len(values[np.where(counts>2)]))

# eigens = np.linalg.eigvalsh(similarity_matrix)
# eigens = np.flip(eigens)

# plt.figure(figsize=(10, 8))
# plt.plot(list(range(1, vocab_size + 1)), eigens + 1e-3, 'o')
# plt.vlines(embedding_dim, 0, 100, 'red')
# #plt.xscale('log')
# plt.yscale('log')
# #plt.show()

# print(similarity_matrix.shape)
# print(np.linalg.matrix_rank(similarity_matrix))

# plt.figure(figsize=(10, 8))
# sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels, cmap='viridis')
# plt.xticks(fontsize=7)
# plt.yticks(fontsize=7)
# plt.title('Dot Product Similarity of Normalized Linspaced Embeddings')
#plt.show()


seed = 42
num_samples = 100
sequence_length = 10
context_window = 10
vocab_size = round(sequence_length * 7.8125)
vocab = list(range(vocab_size))
embedding_dim = 50
distr_param = 1.1


rs = np.random.RandomState(seed)
        
num_samples = num_samples  # number of samples
seq_len = sequence_length  # length of the sequences
vocab = vocab  # list of the vocabulary
vocab_size = len(vocab)  # size of the vocabulary
embedding_dim = embedding_dim  # dimension in the embedding space
n_classes = vocab_size + 1  # GPT predicts a token from the vocabulary plus the <EOS>
temperature = 2

# Shuffle the order of the vocabulary
rs.shuffle(vocab)

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
        #step = num_rows // self.vocab_size
        step = 1
        for i, line in enumerate(lines):
            if i >= vocab_size * step:  # Break if enough vectors are read
                break
            if i % step == 0:  # Only take every step-th vector
                values = line.split()
                vector = torch.tensor([float(val) for val in values[1:]])
                embedding_vectors.append(vector)
            
    embedding_matrix = embedding_vectors

    print(len(B), len(embedding_matrix),  len(B[0]), len(embedding_matrix[0]), type(B), type(embedding_matrix), type(B[0]), type(embedding_matrix[0]))
    
X = []  # List for the design matrix
n_gen_seqs = 0  # Total Number of generated sequences

# Process for standard positional encoding as Attention is All You Need
max_pos = context_window
#position_enc = torch.tensor([[torch.sin(torch.tensor(pos / (10000 ** (i // 2 * 2.0 / self.embedding_dim)), dtype=torch.float)) if i % 2 == 0 else torch.cos(torch.tensor(pos / (10000 ** (i // 2 * 2.0 / self.embedding_dim)), dtype=torch.float)) for i in range(self.embedding_dim)] for pos in range(max_pos)], dtype=torch.float)
position_enc = torch.tensor([[1 - (0.5 * pos / (max_pos - 1)) for _ in range(embedding_dim)] for pos in range(max_pos - 1, -1, -1)], dtype=torch.float)
#stuck_limit = self.seq_len * 5  # Number of iterations that determines if the sequence is cursed, and hence should be dropped

# Loop to build the num_samples sequences

pbar = tqdm(total=num_samples, desc='Generating dataset', unit='sample')  # Initialize progress bar

while(n_gen_seqs < num_samples):

    sequence = []  # Initialise the seq to an empty list
    length = 0  # Variable for the length of the sequence
    #stuck = 0  # Counter that establish if the algorithm in stuck
    
    while(length < seq_len):  # While loop that generates the sequence
        
        
        # If it is the first token, then sample it uniformily from the vocabulary
        if length == 0:
            
            while True:
                # Sample from the distribution
                number = rs.zipf(distr_param)

                # Check if the number is within the range
                if number < vocab_size:
                    break  # Exit the loop if the number is within the range
            
            sequence.append(vocab[number - 1])
            length += 1
            

        else:  #if it is another token then choose it from similarity
            
            # Combine token and positional embeddings 
            combined_embedding = torch.zeros_like(embedding_matrix[0])
            j = context_window-1
            for i in range(length - 1, max(length - context_window - 1, -1), -1):
                token_index = vocab.index(sequence[i])
                combined_embedding += embedding_matrix[token_index] * position_enc[j]
                j -= 1
                    
                
            #combined_embedding = combined_embedding + position_enc[length - 1]

            # Calculate similarity with previous tokens and select the most similar one
            similarities = [torch.dot(embedding_matrix[k], combined_embedding) for k in range(vocab_size)]
            similarities = torch.tensor([similarities]) / temperature
            probs = nn.functional.softmax(similarities[0], dim=0)
            probs = probs.numpy()

            # for k in range(length):
            #     probs[vocab.index(sequence[k])] = 0

            #probs /= probs.sum()
        
            next_token = rs.choice(vocab, p=probs)
                
            sequence.append(next_token)
            length += 1
                        
        # stuck += 1
        
        # # I assumed that if took more then stuck limit iterations to build the sequence, then the seq is cursed
        # if(stuck == stuck_limit):
        #     stuck = 0
        #     sequence = []
        #     length = 0
            
            
    # Check if the built sequence is already in the dataset
    if(n_gen_seqs != 0):
        is_in_matrix = sequence in X
    else:
        is_in_matrix = False  # If it is the first sequence add it in X
    
    
    # If the generated sequence is not already present, build the padded seqs 
    if(not is_in_matrix):
        
        X.append(sequence)
        n_gen_seqs += 1
        pbar.update(1)

pbar.close()
print("\n")
X_1 = np.array(X)

#y = np.hstack((X[:, 1:], np.full((X[:, 1:].shape[0], 1), vocab_size)))  # shift target sequence to the right



# distr = []

# # Count how many times a token appears
# for i in (vocab):
#     z = len(X[np.where(X == i)])
#     distr.append(z)

# # Compute the frequencies of each token
# num_tok = X.size
# distr = np.array(distr) / num_tok

# # Sort the frequencies in descending order
# distr[::-1].sort()
# distr = distr.tolist()

# plt.figure(figsize=(12, 8))
# plt.plot(range(1, vocab_size + 1), distr, '-', color = 'blue', linewidth=2, label = 'Observed distribution')

# x_values = np.linspace(1, vocab_size, vocab_size)  # Generating x values for the function
# y_values = np.max(distr) / (x_values + 0) ** (1)


# plt.plot(x_values, y_values, color='skyblue', linewidth=2, label = 'Zipf`s Law: k = 1')
# plt.xlabel('Degree', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# #plt.xscale('log')
# #plt.yscale('log')
# plt.legend(fontsize=14)
# plt.title('Distribution of the tokens', fontsize=16)
# #plt.show()
# plt.clf()
# plt.close()

# # Define the function to fit
# def func(x, a, b):
#     return  1 / (x + b) ** a

# # Plot the data
# plt.figure(figsize=(12, 8))
# plt.plot(x_values, distr, '-', color='blue', linewidth=3, label = 'Observed distribution (normal)' )

# # Fit the function to the data
# popt, pcov = curve_fit(func, x_values, distr, (1, 2.7))

# # Plot the fitted function
# plt.plot(x_values, func(x_values, *popt), color='skyblue', linewidth=3, label=f'Zipf-Mandelbrot Law:  k={popt[0]:.2f}, {popt[1]:.2f}')

# # Plot settings
# plt.xlabel('Degree', fontsize=16)
# plt.ylabel('Frequency', fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# #plt.xscale('log')
# #plt.yscale('log')
# plt.title('Distribution of the tokens', fontsize=18)
# plt.legend(fontsize=15)
# plt.show()



rs = np.random.RandomState(seed)
        
num_samples = num_samples  # number of samples
seq_len = sequence_length  # length of the sequences
vocab = vocab  # list of the vocabulary
vocab_size = len(vocab)  # size of the vocabulary
embedding_dim = embedding_dim  # dimension in the embedding space
n_classes = vocab_size + 1  # GPT predicts a token from the vocabulary plus the <EOS>
temperature = 2

# Shuffle the order of the vocabulary
rs.shuffle(vocab)

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
        #step = num_rows // self.vocab_size
        step = 1
        for i, line in enumerate(lines):
            if i >= vocab_size * step:  # Break if enough vectors are read
                break
            if i % step == 0:  # Only take every step-th vector
                values = line.split()
                vector = torch.tensor([float(val) for val in values[1:]])
                embedding_vectors.append(vector)
            
    embedding_matrix = embedding_vectors
    embedding_matrix = B
    
X = []  # List for the design matrix
n_gen_seqs = 0  # Total Number of generated sequences

# Process for standard positional encoding as Attention is All You Need
max_pos = context_window
#position_enc = torch.tensor([[torch.sin(torch.tensor(pos / (10000 ** (i // 2 * 2.0 / self.embedding_dim)), dtype=torch.float)) if i % 2 == 0 else torch.cos(torch.tensor(pos / (10000 ** (i // 2 * 2.0 / self.embedding_dim)), dtype=torch.float)) for i in range(self.embedding_dim)] for pos in range(max_pos)], dtype=torch.float)
position_enc = torch.tensor([[1 - (0.5 * pos / (max_pos - 1)) for _ in range(embedding_dim)] for pos in range(max_pos - 1, -1, -1)], dtype=torch.float)
#stuck_limit = self.seq_len * 5  # Number of iterations that determines if the sequence is cursed, and hence should be dropped

# Loop to build the num_samples sequences

pbar = tqdm(total=num_samples, desc='Generating dataset', unit='sample')  # Initialize progress bar

while(n_gen_seqs < num_samples):

    sequence = []  # Initialise the seq to an empty list
    length = 0  # Variable for the length of the sequence
    #stuck = 0  # Counter that establish if the algorithm in stuck
    
    while(length < seq_len):  # While loop that generates the sequence
        
        
        # If it is the first token, then sample it uniformily from the vocabulary
        if length == 0:
            
            while True:
                # Sample from the distribution
                number = rs.zipf(distr_param)

                # Check if the number is within the range
                if number < vocab_size:
                    break  # Exit the loop if the number is within the range
            
            sequence.append(vocab[number - 1])
            length += 1
            

        else:  #if it is another token then choose it from similarity
            
            # Combine token and positional embeddings 
            combined_embedding = torch.zeros_like(embedding_matrix[0])
            j = context_window-1
            for i in range(length - 1, max(length - context_window - 1, -1), -1):
                token_index = vocab.index(sequence[i])
                combined_embedding += embedding_matrix[token_index] * position_enc[j]
                j -= 1
                    
                
            #combined_embedding = combined_embedding + position_enc[length - 1]

            # Calculate similarity with previous tokens and select the most similar one
            similarities = [torch.dot(embedding_matrix[k], combined_embedding) for k in range(vocab_size)]
            similarities = torch.tensor([similarities]) / temperature
            probs = nn.functional.softmax(similarities[0], dim=0)
            probs = probs.numpy()

            # for k in range(length):
            #     probs[vocab.index(sequence[k])] = 0

            #probs /= probs.sum()
        
            next_token = rs.choice(vocab, p=probs)
                
            sequence.append(next_token)
            length += 1
                        
        # stuck += 1
        
        # # I assumed that if took more then stuck limit iterations to build the sequence, then the seq is cursed
        # if(stuck == stuck_limit):
        #     stuck = 0
        #     sequence = []
        #     length = 0
            
            
    # Check if the built sequence is already in the dataset
    if(n_gen_seqs != 0):
        is_in_matrix = sequence in X
    else:
        is_in_matrix = False  # If it is the first sequence add it in X
    
    
    # If the generated sequence is not already present, build the padded seqs 
    if(not is_in_matrix):
        
        X.append(sequence)
        n_gen_seqs += 1
        pbar.update(1)

pbar.close()
print("\n")
X = np.array(X)
y = np.hstack((X[:, 1:], np.full((X[:, 1:].shape[0], 1), vocab_size)))  # shift target sequence to the right


distr = []

# Count how many times a token appears
for i in (vocab):
    z = len(X[np.where(X == i)])
    distr.append(z)

# Compute the frequencies of each token
num_tok = X.size
distr = np.array(distr) / num_tok

# Sort the frequencies in descending order
distr[::-1].sort()
distr = distr.tolist()

distr_1 = []

# Count how many times a token appears
for i in (vocab):
    z = len(X_1[np.where(X_1 == i)])
    distr_1.append(z)

# Compute the frequencies of each token
num_tok = X_1.size
distr_1 = np.array(distr_1) / num_tok

# Sort the frequencies in descending order
distr_1[::-1].sort()
distr_1 = distr_1.tolist()


# plt.figure(figsize=(12, 8))
# plt.plot(range(1, vocab_size + 1), distr, '-', color = 'blue', linewidth=2, label = 'Observed distribution')

x_values = np.linspace(1, vocab_size, vocab_size)  # Generating x values for the function
y_values = np.max(distr) / (x_values + 0) ** (1)
y_values_1 = np.max(distr_1) / (x_values + 0) ** (1)


plt.plot(x_values, y_values, color='skyblue', linewidth=2, label = 'Zipf`s Law: k = 1')
plt.xlabel('Degree', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
#plt.xscale('log')
#plt.yscale('log')
plt.legend(fontsize=14)
plt.title('Distribution of the tokens', fontsize=16)
#plt.show()
plt.clf()
plt.close()

# Define the function to fit
def func(x, a, b):
    return  1 / (x + b) ** a

# Plot the data
plt.figure(figsize=(12, 8))
plt.plot(x_values, distr, '-', color='blue', linewidth=3, label = 'Observed distribution (generated)' )
plt.plot(x_values, distr_1, '-', color='red', linewidth=3, label = 'Observed distribution (original)' )

# Fit the function to the data
popt, pcov = curve_fit(func, x_values, distr, (1, 2.7))

# Plot the fitted function
plt.plot(x_values, func(x_values, *popt), color='skyblue', linewidth=3, label=f'Zipf-Mandelbrot Law:  k={popt[0]:.2f}, {popt[1]:.2f}')

# Plot settings
plt.xlabel('Degree', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.xscale('log')
#plt.yscale('log')
plt.title('Distribution of the tokens', fontsize=18)
plt.legend(fontsize=15)
plt.show()