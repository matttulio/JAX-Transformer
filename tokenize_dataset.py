from datasets import load_dataset
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK tokenizer
nltk.download('punkt_tab')

# Load the IMDb dataset from Hugging Face
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Tokenize and count word frequencies
token_counter = Counter()

# Tokenize each review in the training set
for example in dataset['train']:
    review = example['text']
    tokens = word_tokenize(review.lower())  # Tokenize the review into words
    token_counter.update(tokens)


# Save only the top N frequencies to a txt file (without words)
with open("Datasets/real_freqs.txt", "w") as f:
    for token, count in token_counter.most_common():
        f.write(f"{token}: {count}\n")


