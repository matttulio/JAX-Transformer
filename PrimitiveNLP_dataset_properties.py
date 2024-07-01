import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pickle
import os
import seaborn as sns
import torch
from src.benchmarks import PrimitiveNLP
from scipy.optimize import curve_fit
from scipy.stats import chi2_contingency

plt.style.use('science')

print("\n")
print("PROPERTIES OF THE PRIMITIVE NLP DATASET")
print("\n")
data_dir = 'Datasets/Data'
file_name = 'primitive_NLP_dataset_n_smpl200000__seq_len10__cont_win10__'\
        'v_size78__emb_dim50__emb_typeglove.6B.50d__seed42__d_par1.1__temp2.pkl'

data_path = os.path.join(data_dir, file_name)

with open(data_path, "rb") as file: 
    dataset = pickle.load(file)


# ============== Dot product similarity of tokens ============== #
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

plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 0

embedding_path = 'Datasets/glove/glove.6B.50d.txt'
vocab_size = dataset.vocab_size  # Adjust this based on your requirements

embeddings = read_logspaced_embeddings(embedding_path, vocab_size)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms
similarity_matrix = np.dot(embeddings, embeddings.T)
labels = [f'{i+1}' for i in range(vocab_size)]

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels, cmap='viridis')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.title('Dot Product Similarity of Normalized Logspaced Embeddings')
plt.savefig('Datasets/Data/Properties/similarities_logspaced.pdf', bbox_inches='tight')
plt.close()

embeddings = read_embeddings(embedding_path, vocab_size)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms
similarity_matrix = np.dot(embeddings, embeddings.T)

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels, cmap='viridis')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.title('Dot Product Similarity of Normalized Linspaced Embeddings')
plt.savefig('Datasets/Data/Properties/similarities_linspaced.pdf', bbox_inches='tight')
plt.close()


# ============== Dataset statistics ============== #
unique_rows, counts = np.unique(dataset.X, axis=0, return_counts=True)
unique_row_counts = dict(zip(map(tuple, unique_rows), counts))

print("Percentage of unique samples: ", (len(unique_row_counts) / dataset.num_samples) * 100, "\n")

print("Statistical properties of the dataset:")
print(f"Mean of the tokens: {np.mean(dataset.X):.2f}, "
      f"STD of the tokens: {np.std(dataset.X):.2f}, "
      f"Median of the tokens: {np.median(dataset.X):.2f}")
print(f"Mean of the labels: {np.mean(dataset.y):.2f}, "
      f"STD of the labels: {np.std(dataset.y):.2f}")
print("\n")

# ============== Distribution of targets ============== #
distr = []
vocab_y = np.unique(dataset.y)  # Convert y to a set to get unique values, then convert back to list
total_sequences = dataset.y.shape[0] * dataset.y.shape[1]  # Assuming dataset.seq_len represents the length of each sequence

for i in vocab_y:
    h = len(dataset.y[np.where(dataset.y == i)])  # Count occurrences of each value in y
    h = (h / total_sequences) * 100  # Calculate the percentage
    distr.append(h)

plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 2

plt.figure(figsize=(12, 8))
plt.bar(vocab_y, distr, color='skyblue')
plt.xticks([0, 1], ['0', '1'], fontsize=14)
plt.xlabel('Labels', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of the labels', fontsize=16)
plt.savefig('Datasets/Data/Properties/distribution_of_labelsSUM.pdf', bbox_inches='tight')
plt.close()

# ============== Distribution of input tokens ============== #
distr = []
# Count how many times a token appears
for i in (dataset.vocab):
    z = len(dataset.X[np.where(dataset.X == i)])
    distr.append(z)

# Compute the frequencies of each token
num_tok = dataset.X.size
distr = np.array(distr) / num_tok

# Sort the frequencies in descending order
distr[::-1].sort()
distr = distr.tolist()

plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['ytick.minor.size'] = 2

plt.figure(figsize=(12, 8))
plt.plot(range(1, dataset.vocab_size + 1), distr, '-', color = 'blue', label = 'Observed distribution')

x_values = np.linspace(1, dataset.vocab_size, dataset.vocab_size)  # Generating x values for the function
y_values = np.max(distr) / (x_values + 0) ** (1)

plt.plot(x_values, y_values, color='skyblue', label = 'Zipf`s Law: k = 1')
plt.xlabel('Degree', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.title('Distribution of the tokens', fontsize=16)
#plt.show()
plt.clf()
plt.close()

# ============== Fit Zipf's Law ============== #
# Define the function to fit
def func(x, k, b):
    return  1 / (x + b) ** k

# Plot the data
plt.figure(figsize=(12, 8))
plt.plot(x_values, distr, '-', color='blue', linewidth=2, label = 'Observed distribution' )

# Fit the function to the data
popt, pcov = curve_fit(func, x_values, distr, (1, 2.7))

# Plot the fitted function
plt.plot(x_values, func(x_values, *popt), color='skyblue', linewidth=2, label=f'Zipf`s Law:  k={popt[0]:.2f}, {popt[1]:.2f}')

# Plot settings
plt.xlabel('Degree', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of the tokens', fontsize=16)
plt.legend(fontsize=14)
plt.savefig('Datasets/Data/Properties/distribution_of_tokensSUM.pdf', bbox_inches='tight')
plt.close()

print("Fit the distribution parameters")
for i, param in enumerate(popt):
    print(f"Optimal parameter {i+1}: {param:.2f}")


# Compute the fitted values
fitted_values = func(x_values, *popt)
table = np.vstack((distr, fitted_values))

# Calculate the statistic of the plot to determine wether the distribution is Zipfian
res = chi2_contingency(table, correction=False)

print(f"Chi square test: {res.statistic:.3f}, {res.pvalue:.3f}")
print("\n")