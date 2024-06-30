import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pickle
import os
from src.benchmarks import NextHistogramDataset

print("\n PROPERTIES OF THE NEXT HISTOGRAM DATASET \n")

data_dir = 'Datasets/Data'
file_name = 'NextHistogramDataset_n_smpl200000__seq_len10__v_size15__seed42.pkl'

data_path = os.path.join(data_dir, file_name)

with open(data_path, "rb") as file: 
    dataset = pickle.load(file)

plt.style.use('science')

plt.rcParams['xtick.major.size'] = 10  
plt.rcParams['ytick.major.size'] = 10  
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 5


unique_rows, counts = np.unique(dataset.X, axis=0, return_counts=True)
unique_row_counts = dict(zip(map(tuple, unique_rows), counts))

print("Percentage of unique samples: ", (len(unique_row_counts) / dataset.num_samples) * 100, "\n")

print("Statistical properties of the dataset:")
print(f"Mean of the tokens: {np.mean(dataset.X):.2f}, "
      f"STD of the tokens: {np.std(dataset.X):.2f}, "
      f"Median of the tokens: {np.median(dataset.X):.2f}")

print("\n")

distr = []

vocab_y = np.unique(dataset.y)  # Convert y to a set to get unique values, then convert back to list
total_tokens = dataset.y.size  # Assuming dataset.seq_len represents the length of each sequence

for i in vocab_y:
    h = len(dataset.y[np.where(dataset.y == i)])  # Count occurrences of each value in y
    h = h / total_tokens  # Calculate the frequency
    distr.append(h)

plt.figure(figsize=(12, 8))

plt.bar(vocab_y, distr, color='skyblue')
plt.xlabel('Labels', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('Distribution of the labels', fontsize=18)

#plt.xscale('log') 
plt.yscale('log') 

plt.xticks(fontsize=14) 
plt.yticks(fontsize=14)  

plt.savefig('Datasets/Data/Properties/distribution_of_labels.pdf', bbox_inches='tight')
plt.close()

vocab = np.arange(dataset.vocab_size)
distr = []
dataset.X = np.array(dataset.X)
for i in (vocab):
    z = len(dataset.X[np.where(dataset.X == i)])
    z = z / ((dataset.X.size))
    distr.append(z)

plt.figure(figsize=(12, 8))

plt.bar(vocab, distr, color='skyblue')
plt.xlabel('Tokens', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('Distribution of the tokens', fontsize=18)
plt.savefig('Datasets/Data/Properties/distribution_of_tokens.pdf', bbox_inches='tight')
plt.close()
