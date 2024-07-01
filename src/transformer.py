import jax
from jax import jit, grad, random
import jax.numpy as jnp
from flax import linen as nn
import optax
import math
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
from flax.training import train_state


class DotProductAttention(nn.Module):
    model_dim: int
    attention_input: str = 'both'

    def setup(self):
        """
        Initialize parameters.
        """
        # Check if the attention_input variable makes sense
        if self.attention_input not in ['both', 'only_sem', 'only_pos']:
            raise ValueError

        # Define dimensions to look at
        if self.attention_input == 'both':
            a = self.model_dim
        elif self.attention_input == 'only_sem' or self.attention_input == 'only_pos':
            a = int(self.model_dim / 2)

        self.F = jnp.zeros((self.model_dim, self.model_dim))

        if self.attention_input in ['both', 'only_sem']:
            self.F = self.F.at[jnp.arange(0, a), jnp.arange(0, a)].set(1.0)
        elif self.attention_input in ['both', 'only_pos']:
            self.F = self.F.at[jnp.arange(a, 2 * a), jnp.arange(a, 2 * a)].set(1.0)

        # Define queries, keys, and values
        self.Q = self.param('Q', lambda rng, shape: jax.random.uniform(rng, shape, minval=-math.sqrt(5), maxval=math.sqrt(5)), (self.model_dim, self.model_dim))
        self.K = self.param('K', lambda rng, shape: jax.random.uniform(rng, shape, minval=-math.sqrt(5), maxval=math.sqrt(5)), (self.model_dim, self.model_dim))
        self.V = self.param('V', lambda rng, shape: jax.random.uniform(rng, shape, minval=-math.sqrt(5), maxval=math.sqrt(5)), (self.model_dim, self.model_dim))

    def __call__(self, x):
        """
        Perform computation.
        Args:
        - x (Tensor): Batch of sequences.
        Returns:
        - x (Tensor): Batch of processed sequences
        """
        # Compute the Qs, Ks and Vs
        Qx = jnp.matmul(jnp.matmul(x, self.F), self.Q)
        Kx = jnp.matmul(jnp.matmul(x, self.F), self.K)
        Vx = jnp.matmul(x, self.V)

        # Compute the attention scores
        attn_scores = jnp.matmul(Qx, jnp.swapaxes(Kx, -1, -2)) / math.sqrt(self.model_dim)

        # Create an upper triangular mask
        batch_size, seq_length, _ = attn_scores.shape
        mask = jnp.triu(jnp.ones((seq_length, seq_length)), k=1).astype(bool)

        # Fill masked positions with -inf
        attn_scores = jnp.where(mask, -jnp.inf, attn_scores)

        # Compute the attention probabilities
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)

        # Compute the processed sequences
        x = jnp.matmul(attn_probs, Vx)

        return x, attn_probs

    

class LearnedPositionalEncoding(nn.Module):
    d_model: int 
    max_seq_length: int

    def setup(self):

        self.pe = self.param('pe', nn.initializers.normal(stddev=0.1), (1, self.max_seq_length, self.d_model))

    def __call__(self, x):
        
        return jnp.tile(self.pe, (x.shape[0], 1, 1))



class TransformerSeq2Seq(nn.Module):
    vocab_size: int
    d_model: int
    hidden_dimension_fc: int
    n_classes: int
    seq_len: int
    attention_input: list
    plotting: bool = False

    def setup(self):
        # Check if model dim is even
        # if it is not, then some passages 
        # in the forward pass are impossible
        if self.d_model % 2 == 1:
            raise ValueError()

        # Create semantic and positional embeddings layers
        self.semantic_emb = nn.Embed(self.vocab_size, self.d_model)
        self.positional_emb = LearnedPositionalEncoding(self.d_model, self.seq_len)

        self.attention = DotProductAttention(self.d_model, attention_input=self.attention_input)
        self.norm = nn.LayerNorm()
        self.fc1 = nn.Dense(self.hidden_dimension_fc)
        self.fc2 = nn.Dense(self.n_classes)

    def __call__(self, x):

        x_sem = self.semantic_emb(x)
        x_pos = self.positional_emb(x)

        # Apply masking if attention_input is set to 'only_sem' or 'only_pos'
        if self.attention_input in ['only_sem', 'only_pos']:
            # Set zeros on the first self.model_dim / 2 entries
            x_sem = x_sem.at[..., int(self.d_model / 2):].set(0.0)
            # Set zeros on the last self.model_dim / 2 entries
            x_pos = x_pos.at[..., :int(self.d_model / 2)].set(0.0)
        

        # Combine semantic and positional embeddings
        x = x_sem + x_pos

        # Apply the transformer architecture
        a, attn_probs = self.attention(x)
        x = self.norm(a)
        x = self.fc2(nn.relu(self.fc1(x)))

        if(self.plotting):
            return x, attn_probs

        return x
    
######################################################
#
# TRAIN AND EVALUATE
#
######################################################


def number_of_parameters(params):
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    return param_count


def eval_init(train_dataset, eval_dataset, state):

    # Get initial metrics before training starts
    print("Initial metrics before training...")

    # ============== Initial Training Metrics ============== #
    # Process only the first batch for initial training metrics
    train_datagen = iter(train_dataset)
    batch = next(train_datagen)
    metrics = eval_step(state, batch)
    print(f"Initial Train loss = {metrics['loss']:.4f}")
    first_train_metrics = metrics

    # ============== Initial Validation Metrics ============== #
    # Process only the first batch for initial validation metrics
    eval_datagen = iter(eval_dataset)
    batch = next(eval_datagen)
    metrics = eval_step(state, batch)
    print(f"Initial Val loss = {metrics['loss']:.4f}, Initial Val accuracy = {metrics['accuracy']:.4f}")
    first_eval_metrics = metrics
    print('\n')

    return first_train_metrics, first_eval_metrics


def predict(dataset, state):
    print(f"Predicting sequences...")
    num_batches = len(dataset)
    datagen = iter(dataset)

    predictions = []
    for batch_idx in tqdm(range(1, num_batches + 1)):

        batch = next(datagen)
        batch, _ = batch
        logits = state.apply_fn(state.params, batch)
        probs = nn.softmax(logits)
        prediction = jnp.argmax(probs, axis=-1)
        predictions.append(prediction)
    
    return predictions

def train_and_evaluate(train_dataset, eval_dataset, state, epochs):

    num_train_batches = len(train_dataset)
    num_eval_batches = len(eval_dataset)

    train_epoch_metrics = []
    val_epoch_metrics = []

    train_batch_metrics = []
    eval_batch_metrics = []
    
    step_idx = 0
    for epoch in range(1, epochs + 1):
        #best_eval_loss = 1e6
        print(f"Epoch {epoch}...")
        
        # ============== Training ============== #
        train_datagen = iter(train_dataset)

        for _ in tqdm(range(1, num_train_batches + 1)):
            batch = next(train_datagen)
            state, metrics = train_step(state, batch)
            train_batch_metrics.append(metrics)

            if(step_idx % 10 == 0):
                eval_datagen = iter(eval_dataset)
                for _ in range(1, num_eval_batches + 1):
                    batch = next(eval_datagen)
                    metrics = eval_step(state, batch)
                    eval_batch_metrics.append(metrics)

            step_idx += 1

        print(f"Train loss = {metrics['loss']:.4f}")
        train_epoch_metrics.append(metrics)

        # ============== Validation ============= #
        last_batch = eval_dataset[-1]
        metrics = eval_step(state, last_batch)
        print(f"Val loss = {metrics['loss']:.4f}, Val accuracy = {metrics['accuracy']:.4f}")
        val_epoch_metrics.append(metrics)

        print("\n")

    if(len(eval_batch_metrics) % 10 != 0):
        eval_batch_metrics.append(val_epoch_metrics[-1])

    return state, train_batch_metrics, eval_batch_metrics, train_epoch_metrics, val_epoch_metrics

@jax.jit
def train_step(state: train_state.TrainState, batch: jnp.ndarray):
    batch, label = batch

    def loss_fn(params):
        logits = state.apply_fn(params, batch)
        loss = cross_entropy_loss(logits=logits, labels=label)
        return loss, logits


    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=label)
    return state, metrics

def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


@jax.jit
def eval_step(state, batch):
    batch, label = batch
    logits = state.apply_fn(state.params, batch)
    return compute_metrics(logits=logits, labels=label)


def cross_entropy_loss(*, logits, labels):
    one_hot_encoded_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    return optax.softmax_cross_entropy(
        logits=logits, labels=one_hot_encoded_labels
    ).mean()

def init_train_state(
    model, random_key, shape, learning_rate, optimizer=optax.adam
) -> train_state.TrainState:
    # Initialize the Model
    variables = model.init(random_key, shape)
    # Create the Optimizer
    optimizer = optimizer(learning_rate)
    # Return a TrainState
    return train_state.TrainState.create(
        apply_fn = model.apply,
        tx = optimizer,
        params = variables
    )

def convert_batch_to_jax(batch):
    jax_batch = []
    for tensor in batch:
        jax_batch.append(jnp.array(tensor.numpy()))
    return jax_batch



######################################################
#
# REPARAMETRIZE
#
######################################################

def reparameterize(vocab_size, model_dim, hidden_dimension_fc, n_classes, seq_len, state, attention_input, dummy_input, learning_rate, rng):
    """
    Function for reparameterizing a transformer model.

    Args:
    - vocab_size (int): Size of the vocabulary.
    - model_dim (int): Dimensionality of the model.
    - hidden_dimension_fc (int): Hidden dimension for the fc layer.
    - n_classes (int): Number of classes.
    - seq_len (int): Length of input sequences.
    - saved_params: Parameters of the original transformer model.

    Returns:
    - new_transformer: Reparameterized transformer model.
    """
    
    # Set the parameters of the new transformer model
    new_transformer_params = state.params

    F = jnp.zeros((model_dim, model_dim))

    if attention_input == 'both':
        a = model_dim
    elif attention_input == 'only_sem' or attention_input == 'only_pos':
        a = int(model_dim / 2)

    if attention_input in ['both', 'only_sem']:
        F = F.at[jnp.arange(0, a), jnp.arange(0, a)].set(1.0)
    elif attention_input in ['both', 'only_pos']:
        F = F.at[jnp.arange(a, 2 * a), jnp.arange(a, 2 * a)].set(1.0)
    
    

    # Reparameterize the attention mechanism
    new_transformer_params['params']['attention']['Q'] = jnp.dot(F, new_transformer_params['params']['attention']['Q'])
    new_transformer_params['params']['attention']['K'] = jnp.dot(F, new_transformer_params['params']['attention']['K'])

    # Zero out the second half of the semantic embedding weights
    new_transformer_params['params']['semantic_emb']['embedding'].at[..., model_dim//2:].set(0.0)

    # Zero out the first half of the positional embedding weights
    new_transformer_params['params']['positional_emb']['pe'].at[..., :model_dim//2].set(0.0)

    # Add small random noise to Q, K, and the embedding weights
    new_transformer_params['params']['attention']['Q'] += 0.001 * jax.random.normal(rng, new_transformer_params['params']['attention']['Q'].shape)
    new_transformer_params['params']['attention']['K'] += 0.001 * jax.random.normal(rng, new_transformer_params['params']['attention']['K'].shape)
    new_transformer_params['params']['semantic_emb']['embedding'] += 0.001 * jax.random.normal(rng, new_transformer_params['params']['semantic_emb']['embedding'].shape)
    new_transformer_params['params']['positional_emb']['pe'] += 0.001 * jax.random.normal(rng, new_transformer_params['params']['positional_emb']['pe'].shape)

    # Create a new transformer model with the same architecture as the original
    new_transformer = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, seq_len, 'both')
    new_state = init_train_state(new_transformer, rng, dummy_input, learning_rate, optax.sgd)

    new_state = state.replace(params=new_transformer_params)

    return new_state