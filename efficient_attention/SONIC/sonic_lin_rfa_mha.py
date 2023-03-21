import sys
import os

sys.path.append(os.getcwd())

import jax.numpy as jnp
import jax
import flax.linen as nn
from jax.nn.initializers import glorot_normal
from flax.linen import softmax

import flax.linen as nn

import numpy as np

from efficient_attention.RFA.rfa_mha import random_project, load_random_matrices, build_random_matrices, _normalize

import pdb ## For debugging purposes only.

EPS = 1.0
RANDOM_MATRICES_PATH = os.path.join(os.path.dirname(__file__), './RFA_random_matrices')

class MHA(nn.Module):
    hidden_dim : int
    head_dim : int
    num_heads : int
    dropout : float
    mask : bool
    downsampling_k : int = 64
    tau: float = 1.0
    reparam_proj: bool = False

    """
    ## For some reason putting the initializers over here doesn't seem to work.
    ## They are somehow inextricably tied to the other variables defined above.
    ## It may be valuable to figure out why on earth this happens.
    query_kernel_init = jax.nn.initializers.glorot_normal()
    key_kernel_init = jax.nn.initializers.glorot_normal
    value_kernel_init = jax.nn.initializers.glorot_normal
    """
    def setup(self):
        ## Preambulatory work of setting up the initializers and weights.
        init_shape = (self.hidden_dim, self.num_heads, self.head_dim)
        self.query_kernel = self.param('query_kernel', jax.nn.initializers.glorot_uniform(), init_shape, jnp.float32)
        self.key_kernel = self.param('key_kernel', jax.nn.initializers.glorot_uniform(), (self.hidden_dim, self.num_heads, self.head_dim))
        self.value_kernel = self.param('value_kernel', jax.nn.initializers.glorot_uniform(), (self.hidden_dim, self.num_heads, self.head_dim))

        self.numerical_stabilizer = 0.001

        downsampling_shape_128 = (self.downsampling_k, 128)
        downsampling_shape_512 = (self.downsampling_k, 512)
        mean = 0.0
        sd = float(1)/float(self.downsampling_k)

        self.key_downsampling_mat_128 = self.param('key_downsample_mat_128', lambda rng, shape, mean, sd: mean + sd * jax.random.normal(rng, shape=shape), downsampling_shape_128, mean, sd)
        self.key_downsampling_mat_512 = self.param('key_downsample_mat_512', lambda rng, shape, mean, sd: mean + sd * jax.random.normal(rng, shape=shape), downsampling_shape_512, mean, sd)
        self.value_downsampling_mat_128 = self.param('value_downsample_mat_128', lambda rng, shape, mean, sd: mean + sd * jax.random.normal(rng, shape=shape), downsampling_shape_128, mean, sd)
        self.value_downsampling_mat_512 = self.param('value_downsample_mat_512', lambda rng, shape, mean, sd: mean + sd * jax.random.normal(rng, shape=shape), downsampling_shape_512, mean, sd)

        ## Load random matrices.
        self.random_matrices = load_random_matrices(head_dim=self.head_dim, proj_dim=self.hidden_dim)
        if self.reparam_proj:
            self.sigma = self.param('sigma', jax.nn.initializers.constant(1.), (self.num_heads, 1, self.head_dim))

        ## Dropout layers.
        self.dropout_layer = nn.Dropout(0.1)


    def sample_random_matrices(self):
        num_random_matrices = self.random_matrices.shape[0]
        indices = np.random.choice(
            num_random_matrices,
            size=self.num_heads,
            replace=False)
        # [num_heads, proj_dim, head_dim]
        random_matrices = self.random_matrices[indices]
        return random_matrices


    def __call__(self, x, *, train):
        ## Jax complains about passing in multiple arguments.
        ## So we do the hack of concatenating the queries, keys and values into a list and unpacking it.
        query, key, value = x

        assert all(len(i.shape) == 3 for i in x), "Incorrect size of input, should be [batch, seq length, hidden dimension]"
        if value.shape[1] == key.shape[1] == 128:
            key = jnp.einsum('ks, bsd -> bkd', self.key_downsampling_mat_128, key)
            value = jnp.einsum('ks, bsd -> bkd', self.value_downsampling_mat_128, value)
        elif value.shape[1] == key.shape[1] == 512:
            key = jnp.einsum('ks, bsd -> bkd', self.key_downsampling_mat_512, key)
            value = jnp.einsum('ks, bsd -> bkd', self.value_downsampling_mat_512, value)
        else:
            raise Exception("Input sequence length must be of size 128 or 512.")

        ## First, we map the queries keys and values.
        queries = jnp.einsum('bqd, dnh -> bqnh', query, self.query_kernel)
        keys = jnp.einsum('bkd, dnh -> bknh', key, self.key_kernel)
        values = jnp.einsum('bkd, dnh -> bknh', value, self.value_kernel)

        if self.mask:  ## Here, we do normal linformer-style attention.
            ## We have to multiply the queries with the keys.
            q_ks = jnp.einsum('bqnh, bknh -> bnqk', queries, keys)
            trilled_mask = jnp.ones((queries.shape[0], queries.shape[2], queries.shape[1], keys.shape[1])).astype(
                bool)  ## This is of size: [batch_size, num_heads, query_seq_length, key_seq_length]
            trilled_mask = jnp.tril(trilled_mask)
            ## TODO, check for correctness
            trilled_mask = trilled_mask[:, :, :, :self.downsampling_k]
            q_ks = jnp.where(trilled_mask == False, -9e15, q_ks)

            ## Then we take the softmax
            attn_mat = softmax(q_ks)

            attn_mat = self.dropout_layer(attn_mat, deterministic=not train)

            ## Then we right multiply by the values and return the result.
            a_v = jnp.einsum('bhqk, bkhd -> bqhd', attn_mat, values)

        else:
            random_matrices = self.sample_random_matrices()
            # random_matrices = self.random_matrices[0:self.num_heads, ...]

            random_matrices = build_random_matrices(random_matrices=random_matrices,
                                                          tau=self.tau,
                                                          sigma=self.sigma if self.reparam_proj else None,
                                                          reparam_proj=self.reparam_proj)

            phi_k = random_project(x=keys, random_matrices=random_matrices)
            s = jnp.einsum("bknd, bknh -> bndh", phi_k, values)
            z = jnp.sum(phi_k, axis=1)

            phi_q = random_project(x=queries, random_matrices=random_matrices)
            qs = jnp.einsum("bqnd, bndh -> bqnh", phi_q, s)
            qz = jax.lax.clamp(EPS, jnp.abs(jnp.einsum("bqnd, bnd -> bqn", phi_q, z)), 10e9)
            a_v = qs / jnp.expand_dims(qz, axis=-1)

        ## Finally, concatenate across the head dimension.
        return a_v.reshape((a_v.shape[0], a_v.shape[1], a_v.shape[2]*a_v.shape[3]))

"""
## A place to unit test my Multi-Head-Attention Implementation.
## Unit tests are always great!
from jax import random

hidden_dim = 15
head_dim = 5
num_heads = 3
dropout = 0.1
mask = False

batch_size = 2
sequence_length = 128
downsampling_k = 64
mha = MHA(hidden_dim, head_dim, num_heads, dropout, mask, sequence_length, downsampling_k)

param_key = random.PRNGKey(42)
dropout_key = random.PRNGKey(43)
qs = random.uniform(random.PRNGKey(44), (batch_size, sequence_length, hidden_dim))
ks = random.uniform(random.PRNGKey(45), (batch_size, sequence_length, hidden_dim))
vs = random.uniform(random.PRNGKey(46), (batch_size, sequence_length, hidden_dim))
params = mha.init({'params': param_key, 'dropout': dropout_key}, [qs, ks, vs], train=True)
## One thing to keep note is that a new dropout_key must constantly be passed into the function.
attention_mat = mha.apply(params, [qs, ks, vs], train=True, rngs={'dropout': dropout_key})
## Further sanity checks.
print(attention_mat)
"""