import sys
import os

import jax.numpy as jnp
import jax

import flax.linen as nn
from flax.linen.linear import DenseGeneral, default_kernel_init

import functools
from typing import (Any, Callable, Optional, Tuple)

sys.path.append(os.getcwd())

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

import time

import pdb ## For debugging purposes only.

## This is an implementation of the random + dense synthesizer implemntation.

class MHA(nn.Module):
    hidden_dim : int
    head_dim : int
    num_heads : int
    dropout : float
    mask : bool
    sequence_length : int = 128 ## We pretty much only train on this sequence length anyways. 
    dtype: Optional[Dtype] = None
    param_dtype: Any = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    use_bias: bool = True
    precision: nn.linear.PrecisionLike = None
    numerical_stabilizer: float = 0.001
    up_train: bool = False

    """
    ## For some reason putting the initializers over here doesn't seem to work.
    ## They are somehow inextricably tied to the other variables defined above.
    ## It may be valuable to figure out why on earth this happens.
    query_kernel_init = jax.nn.initializers.glorot_normal()
    key_kernel_init = jax.nn.initializers.glorot_normal
    value_kernel_init = jax.nn.initializers.glorot_normal
    """

    def setup(self):
        self.dense_queries = DenseGeneral(axis=-1, dtype=self.dtype, param_dtype=self.param_dtype, features=(self.num_heads, self.head_dim), kernel_init=self.kernel_init,
                                          bias_init=self.bias_init, use_bias=self.use_bias, precision=self.precision, name='query')
        self.dense_keys = DenseGeneral(axis=-1, dtype=self.dtype, param_dtype=self.param_dtype, features=(self.num_heads, self.head_dim), kernel_init=self.kernel_init,
                                          bias_init=self.bias_init, use_bias=self.use_bias, precision=self.precision, name='key')
        self.dense_values = DenseGeneral(axis=-1, dtype=self.dtype, param_dtype=self.param_dtype, features=(self.num_heads, self.head_dim), kernel_init=self.kernel_init,
                                          bias_init=self.bias_init, use_bias=self.use_bias, precision=self.precision, name='value')

        ## For some weird reason, they have a two layer ffn here, instead of a one layer FFN. First layer produces head_dim features, next layer produces seq_length features. ##
        self.dense_transform_one = DenseGeneral(axis=-1, dtype=self.dtype, param_dtype=self.param_dtype, features=self.head_dim, kernel_init=self.kernel_init,
                                                 bias_init=self.bias_init, use_bias=self.use_bias, precision=self.precision, name='dense_transform_one')

        self.dense_transform_two = DenseGeneral(axis=-1, dtype=self.dtype, param_dtype=self.param_dtype, features=self.sequence_length, kernel_init=self.kernel_init,
                                                 bias_init=self.bias_init, use_bias=self.use_bias, precision=self.precision, name='dense_transform_two')

        random_mat_shape = (self.num_heads, self.sequence_length, self.sequence_length)

        mean = 0
        sd = 1
        key = self.make_rng('internal_initializer')
        self.random_mat = mean + sd * jax.random.normal(key, shape=random_mat_shape)

        ## Now we define two scaling factors, required for normalisation. ##
        self.alpha_one = jax.random.normal(key, shape=(1,))
        self.alpha_two = jax.random.normal(key, shape=(1,))

        ## This is for contracting from: (b, s, n, h) back to: (b, s, model_dimension). ##
        self.dense_out = DenseGeneral(features=self.hidden_dim,
                           axis=(-2, -1),
                           kernel_init=self.kernel_init,
                           bias_init=self.bias_init,
                           use_bias=self.use_bias,
                           dtype=self.dtype,
                           param_dtype=self.param_dtype,
                           precision=self.precision,
                           name='out')

        ## Here, we have a dropout layer. ##
        self.dropout_layer = nn.Dropout(0.1)

    def __call__(self, x, step, *, train):
        ## Jax complains about passing in multiple arguments.
        ## So we do the hack of concatenating the queries, keys and values into a list and unpacking it.

        query, _, value = x

        ## We don't implement decoder-type models at all. ##
        if self.mask:
            raise Exception("We have not implemented the Causal Attention Mechanism!")

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        queries, values = (self.dense_queries(query),
                             self.dense_values(value))

        ## First, we compute the softmax of the "random" matrix. ##
        random_mat = nn.softmax(self.random_mat, axis=-1)

        queries = jnp.transpose(queries, [0, 2, 1, 3])

        ## Next, we compute the Dense synthesizer part. ##
        dense_attn = self.dense_transform_one(queries)
        dense_attn = nn.relu(dense_attn)
        dense_attn = self.dense_transform_two(dense_attn)

        ## now, we have to take some average of the two.
        scale_one = self.alpha_one / (self.alpha_one + self.alpha_two)
        scale_two = self.alpha_two / (self.alpha_one + self.alpha_two)

        random_mat = scale_one * random_mat
        dense_attn = scale_two * dense_attn 

        mixed_attn = nn.softmax(random_mat + dense_attn, axis=-1)
        mixed_attn = self.dropout_layer(mixed_attn, deterministic=not train)

        ## We finally multiply by the values. ##
        final_out = jnp.einsum('bhqk, bkhd -> bqhd', mixed_attn, values)

        ## Finally, concatenate across the head dimension.
        final_out = self.dense_out(final_out)
        return final_out 

"""
## A place to unit test my Multi-Head-Attention Implementation.
## Unit tests are always great!
from jax import random

hidden_dim = 15
head_dim = 5
num_heads = 3
dropout = 0.1
mask = False 
batch_size = 1
sequence_length = 4
mha = MHA(hidden_dim, head_dim, num_heads, dropout, mask, sequence_length)

param_key = random.PRNGKey(42)
dropout_key = random.PRNGKey(44)
misc_key_init = random.PRNGKey(45)
misc_key_test = random.PRNGKey(46)
qs = random.uniform(random.PRNGKey(44), (batch_size, sequence_length, hidden_dim))
ks = random.uniform(random.PRNGKey(45), (batch_size, sequence_length, hidden_dim))
vs = random.uniform(random.PRNGKey(46), (batch_size, sequence_length, hidden_dim))
params = mha.init({'params': param_key, 'internal_initializer' : misc_key_init}, [qs, ks, vs], 0,  train=True)
## One thing to keep note is that a new dropout_key must constantly be passed into the function.
attention_mat = mha.apply(params, [qs, ks, vs], 0,  train=True, rngs={'dropout': dropout_key, 'internal_initializer': misc_key_test})
## Further sanity checks.
print(attention_mat)
print(attention_mat.shape)
"""