import sys
import os

import numpy

sys.path.append(os.getcwd())

import jax.numpy as jnp
import jax
import flax.linen as nn
from jax.nn.initializers import glorot_normal
from flax.linen import softmax

import flax.linen as nn

import numpy as np

import pdb ## For debugging purposes only.

EPS = 1.0
RANDOM_MATRICES_PATH = os.path.join(os.path.dirname(__file__), './RFA_random_matrices')

def build_random_matrices(random_matrices, tau: float, sigma=None, reparam_proj=False):
    if reparam_proj:
        random_matrices = sigma * random_matrices
    return random_matrices / tau

def _normalize(x):
    norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    return jnp.divide(x, norm + 1e-3), norm

def random_project(*, x, random_matrices):
    # x: [seq_len, bsz, num_heads, head_dim]
    # random_matrices: [num_heads, proj_dim, head_dim]

    # [1, 1, num_heads, 1]
    x, x_norm = _normalize(x)
    # [seq_len, bsz, num_heads, proj_dim]
    x = jnp.einsum("bshd,hkd->bshk", x, random_matrices)
    x_sin, x_cos = jnp.sin(x), jnp.cos(x)

    # [seq_len, bsz, num_heads, 2 * proj_dim]
    phi_x = jnp.concatenate([x_sin, x_cos], axis=-1) * 0.1
    return phi_x

def load_random_matrices(
        *,
        head_dim: int,
        proj_dim: int):

    # [num_random_matrices, proj_dim, head_dim]
    if os.path.exists(f"{RANDOM_MATRICES_PATH}/{head_dim}_{proj_dim}.npy"):
        random_matrices = jnp.load(
            f"{RANDOM_MATRICES_PATH}/{head_dim}_{proj_dim}.npy")
    else:
        raise FileNotFoundError("No Random Matrices found! Construct with "
                                "$python3 RFA_random_matrices/construct_random_matrices.py"
                                "<rrf/orf> <head_dim> <hidden_dim>.")
    return random_matrices


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


        # self.random_matrices = utils.load_random_matrices()
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
        # [num_layers * num_heads, proj_dim, head_dim]
        random_matrices = self.random_matrices[indices]
        return random_matrices


    def __call__(self, x, *, train):
        ## Jax complains about passing in multiple arguments.
        ## So we do the hack of concatenating the queries, keys and values into a list and unpacking it.
        query, key, value = x

        assert all(len(i.shape) == 3 for i in x), "Incorrect size of input, should be [batch, seq length, hidden dimension]"

        ## First, we map the queries keys and values.
        queries = jnp.einsum('bqd, dnh -> bqnh', query, self.query_kernel)
        keys = jnp.einsum('bkd, dnh -> bknh', key, self.key_kernel)
        values = jnp.einsum('bkd, dnh -> bknh', value, self.value_kernel)

        random_matrices = self.sample_random_matrices()
        # random_matrices = self.random_matrices[0:self.num_heads, ...]

        random_matrices = build_random_matrices(random_matrices=random_matrices,
                                                tau=self.tau,
                                                sigma=self.sigma if self.reparam_proj else None,
                                                reparam_proj=self.reparam_proj)

        phi_k = random_project(x=keys, random_matrices=random_matrices)
        phi_q = random_project(x=queries, random_matrices=random_matrices)

        if self.mask:
            batch_size, sequence_length, _, _ = keys.shape
            phi_k = jnp.einsum("bsnh -> sbnh", phi_k).reshape((sequence_length, batch_size * self.num_heads, -1))
            phi_q = jnp.einsum("bsnh -> sbnh", phi_q).reshape((sequence_length, batch_size * self.num_heads, -1))
            v = jnp.einsum("bsnh -> sbnh", values).reshape((sequence_length, batch_size * self.num_heads, -1))

            s = jnp.einsum("sbh, sbd -> sbhd", phi_k, v)
            s = jnp.cumsum(s, axis=0)
            qs = jnp.einsum("sbhd, sbh -> sbd", s, phi_q)

            z = jnp.cumsum(phi_k, axis=0)
            qz = jax.lax.clamp(EPS, jnp.einsum("sbh, sbh -> sb", phi_q, z), 10e9)

            a_v = qs / jnp.expand_dims(qz, axis=-1)
            a_v = jnp.einsum("sbd -> bsd", a_v).reshape(batch_size, self.sequence_length, self.num_heads * self.head_dim)
            return a_v

        else:
            s = jnp.einsum("bknd, bknh -> bndh", phi_k, values)
            z = jnp.sum(phi_k, axis=1)

            qs = jnp.einsum("bqnd, bndh -> bqnh", phi_q, s)
            qz = jax.lax.clamp(EPS, jnp.abs(jnp.einsum("bqnd, bnd -> bqn", phi_q, z)), 10e9)
            a_v = qs / jnp.expand_dims(qz, axis=-1)
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
print(attention_mat.shape)
"""