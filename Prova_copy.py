import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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
        step=1
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


embedding_path = 'Datasets/glove/glove.6B.300d.txt'
vocab_size = 1000  # Adjust this based on your requirements
embedding_dim = 300
labels = [f'{i+1}' for i in range(vocab_size)]
embeddings = read_embeddings(embedding_path, vocab_size)
#embeddings = nn.Embedding(vocab_size, embedding_dim)  # embedding layer
#embeddings = (embeddings.weight.data).cpu().detach().numpy()


pca = PCA(n_components=0.9, svd_solver='full')
reduced_embeddings = pca.fit_transform(embeddings)

explained_variance = pca.explained_variance_ratio_
for i, variance in enumerate(explained_variance):
    print(f"Principal Component {i+1}: {variance:.4f}")

# Plot the results
plt.figure(figsize=(12, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], marker='o', edgecolor='b', facecolor='none', alpha=0.5)

for i, label in enumerate(labels):
    plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8)

plt.title('2D PCA of Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()


# plt.figure(figsize=(10, 8))
# sns.heatmap(embeddings, yticklabels=labels, cmap='viridis')
# plt.xticks(fontsize=7)
# plt.yticks(fontsize=7)
# plt.show()

unique_columns = np.unique(embeddings, axis=0)
print(embeddings[0], len(embeddings[0]))
num_unique_columns = unique_columns.shape[0]
print("Number of unique columns:", num_unique_columns)

norms = np.linalg.norm(embeddings, axis=0, keepdims=True)

sorted_indices = np.argsort(-norms, axis=1).flatten()
embeddings = embeddings[:,sorted_indices]

norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms
similarity_matrix = np.dot(embeddings, embeddings.T)

#masked_matrix = similarity_matrix.astype(float)
# np.fill_diagonal(masked_matrix, np.nan)

# Flatten the masked matrix and remove np.nan values
#flattened_masked_matrix = masked_matrix.flatten()
#non_diagonal_elements = flattened_masked_matrix[~np.isnan(flattened_masked_matrix)]

# Get unique values and their counts
values, counts = np.unique(similarity_matrix, return_counts=True)

# Print the unique values and their counts
print("Unique values (excluding diagonal):", values[np.where(counts>2)])
print("Counts:", counts[np.where(counts>2)])
print("Number of unique values (excluding diagonal):", len(values[np.where(counts>2)]))

eigens = np.linalg.eigvalsh(similarity_matrix)
eigens = np.flip(eigens)

plt.figure(figsize=(10, 8))
plt.plot(list(range(1, vocab_size + 1)), eigens + 1e-3, 'o')
plt.vlines(embedding_dim, 0, 100, 'red')
#plt.xscale('log')
plt.yscale('log')
plt.show()

print(similarity_matrix.shape)
print(np.linalg.matrix_rank(similarity_matrix))

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels, cmap='viridis')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.title('Dot Product Similarity of Normalized Linspaced Embeddings')
plt.show()