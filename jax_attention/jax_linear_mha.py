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


class MHA(nn.Module):
    hidden_dim : int
    head_dim : int
    num_heads : int
    dropout : float
    mask : bool
    downsampling_k : int = 64
    eps: float = 1e-6

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

        # self.random_matrices = utils.load_random_matrices()
        self.elu_feature_map = lambda x: nn.elu(x) + 1

        ## Dropout layers.
        self.dropout_layer = nn.Dropout(0.1)


    def __call__(self, x, *, train):
        ## Jax complains about passing in multiple arguments.
        ## So we do the hack of concatenating the queries, keys and values into a list and unpacking it.
        query, key, value = x

        assert all(len(i.shape) == 3 for i in x), "Incorrect size of input, should be [batch, seq length, hidden dimension]"

        ## First, we map the queries keys and values.
        queries = jnp.einsum('bqd, dnh -> bqnh', query, self.query_kernel)
        keys = jnp.einsum('bkd, dnh -> bknh', key, self.key_kernel)
        values = jnp.einsum('bkd, dnh -> bknh', value, self.value_kernel)

        phi_q = self.elu_feature_map(queries)
        phi_k = self.elu_feature_map(keys)

        if self.mask:
            sequence_length, batch_size, _, _ = keys.shape
            phi_k = jnp.einsum("bsnh -> sbnh", phi_k).reshape((sequence_length, batch_size * self.num_heads, -1))
            phi_q = jnp.einsum("bsnh -> sbnh", phi_q).reshape((sequence_length, batch_size * self.num_heads, -1))
            v = jnp.einsum("bsnh -> sbnh", values).reshape((sequence_length, batch_size * self.num_heads, -1))

            s = jnp.einsum("sbh, sbd -> sbhd", phi_k, v)
            s = jnp.cumsum(s, axis=0)
            qs = jnp.einsum("sbhd, sbh -> sbd", s, phi_q)

            z = jnp.cumsum(phi_k, axis=0)
            qz = jnp.einsum("sbh, sbh -> sb", phi_q, z) + self.eps

            a_v = qs / jnp.expand_dims(qz, axis=-1)
            a_v = jnp.einsum("sbd -> bsd", a_v).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
            return a_v

        else:
            s = jnp.einsum("bknd, bknh -> bndh", phi_k, values)
            z = jnp.sum(phi_k, axis=1)

            qs = jnp.einsum("bqnd, bndh -> bqnh", phi_q, s)
            qz = jnp.einsum("bqnd, bnd -> bqn", phi_q, z) + self.eps

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
mask = True

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