import sys
import os

sys.path.append(os.getcwd())

import jax.numpy as jnp
import jax
from jax.nn.initializers import glorot_normal
from flax.linen import softmax

import flax.linen as nn
from flax.linen.linear import DenseGeneral, default_kernel_init
from typing import (Any, Callable, Optional, Tuple)
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

import numpy as np

import pdb ## For debugging purposes only.

class MHA(nn.Module):
    hidden_dim : int
    head_dim : int
    num_heads : int
    dropout : float
    mask : bool
    downsampling_k : int = 64
    dtype: Optional[Dtype] = None
    param_dtype: Any = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    use_bias: bool = True
    precision: nn.linear.PrecisionLike = None
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
        self.numerical_stabilizer = 0.001

        downsampling_shape_128 = (self.downsampling_k, 128)
        downsampling_shape_512 = (self.downsampling_k, 512)
        mean = 0.0
        sd = float(1)/float(self.downsampling_k)

        self.key_downsampling_mat_128 = self.param('key_downsample_mat_128', lambda rng, shape, mean, sd: mean + sd * jax.random.normal(rng, shape=shape), downsampling_shape_128, mean, sd)
        self.key_downsampling_mat_512 = self.param('key_downsample_mat_512', lambda rng, shape, mean, sd: mean + sd * jax.random.normal(rng, shape=shape), downsampling_shape_512, mean, sd)
        self.value_downsampling_mat_128 = self.param('value_downsample_mat_128', lambda rng, shape, mean, sd: mean + sd * jax.random.normal(rng, shape=shape), downsampling_shape_128, mean, sd)
        self.value_downsampling_mat_512 = self.param('value_downsample_mat_512', lambda rng, shape, mean, sd: mean + sd * jax.random.normal(rng, shape=shape), downsampling_shape_512, mean, sd)


        self.dense_queries = DenseGeneral(axis=-1, dtype=self.dtype, param_dtype=self.param_dtype, features=(self.num_heads, self.head_dim), kernel_init=self.kernel_init,
                                          bias_init=self.bias_init, use_bias=self.use_bias, precision=self.precision, name='query')
        self.dense_keys = DenseGeneral(axis=-1, dtype=self.dtype, param_dtype=self.param_dtype, features=(self.num_heads, self.head_dim), kernel_init=self.kernel_init,
                                          bias_init=self.bias_init, use_bias=self.use_bias, precision=self.precision, name='key')
        self.dense_values = DenseGeneral(axis=-1, dtype=self.dtype, param_dtype=self.param_dtype, features=(self.num_heads, self.head_dim), kernel_init=self.kernel_init,
                                          bias_init=self.bias_init, use_bias=self.use_bias, precision=self.precision, name='value')
        self.dense_out = DenseGeneral(features=self.hidden_dim,
                           axis=(-2, -1),
                           kernel_init=self.kernel_init,
                           bias_init=self.bias_init,
                           use_bias=self.use_bias,
                           dtype=self.dtype,
                           param_dtype=self.param_dtype,
                           precision=self.precision,
                           name='out')


        ## Dropout layers.
        self.dropout_layer = nn.Dropout(0.1)

    def __call__(self, x, switch: bool, *, train): 
        ## Jax complains about passing in multiple arguments.
        ## So we do the hack of concatenating the queries, keys and values into a list and unpacking it.
        query, key, value = x
        assert all(len(i.shape) == 3 for i in x), "Incorrect size of input, should be [batch, seq length, hidden dimension]"
        if key.shape[1] == value.shape[1] == 128 and self.up_train and switch:
            key = jnp.einsum('ks, bsd -> bkd', self.key_downsampling_mat_128, key)
            value = jnp.einsum('ks, bsd -> bkd', self.value_downsampling_mat_128, value)
        elif query.shape[1] == value.shape[1] == 512 and self.up_train and switch:
            key = jnp.einsum('ks, bsd -> bkd', self.key_downsampling_mat_512, key)
            value = jnp.einsum('ks, bsd -> bkd', self.value_downsampling_mat_512, value)

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        queries, keys, values = (self.dense_queries(query),
                                 self.dense_keys(key),
                                 self.dense_values(value))

        if self.mask: ## Here, we do normal linformer-style attention.
            ## We have to multiply the queries with the keys.
            q_ks = jnp.einsum('bqnh, bknh -> bnqk', queries, keys)
            trilled_mask = jnp.ones((queries.shape[0], queries.shape[2], queries.shape[1], keys.shape[1])).astype(bool) ## This is of size: [batch_size, num_heads, query_seq_length, key_seq_length]
            trilled_mask = jnp.tril(trilled_mask)
            ## TODO, check for correctness
            trilled_mask = trilled_mask[:, :, :, :self.downsampling_k] 
            q_ks = jnp.where(trilled_mask == False, -9e15, q_ks)

            ## Then we take the softmax
            attn_mat = softmax(q_ks)

            attn_mat = self.dropout_layer(attn_mat, deterministic=not train)

            ## Then we right multiply by the values and return the result.
            a_v =  jnp.einsum('bhqk, bkhd -> bqhd', attn_mat, values)
        else:
            ## Non-causal numerator. ##
            ## The following is taken directly from the Performer Code. ##
            ## source: https://github.com/google-research/google-research/blob/master/performer/fast_attention/tensorflow/fast_attention.py#L322 ##

            ## We then re-transform the queries and the keys.
            queries = nn.relu(queries) + self.numerical_stabilizer
            keys = nn.relu(keys) + self.numerical_stabilizer

            ## TODO, check for correctness. ##
            kvs = jnp.einsum("blhm,blhd->bhmd", keys, values)
            a_v = jnp.einsum("blhm,bhmd->blhd", queries, kvs)
            ## Non-causal denominator. ##
            ks_sum = jnp.einsum("blhm->bhm", keys)
            normalizer = jnp.einsum("blhm,bhm->blh", queries, ks_sum)
            ## Then we transpose back and do the normalization. ##
            normalizer = jnp.expand_dims(normalizer, len(normalizer.shape))
            a_v = a_v / normalizer

        ## Finally, concatenate across the head dimension.
        out = self.dense_out(a_v)
        return out

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
sequence_length = 4
downsampling_k = 2
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
