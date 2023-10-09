import sys
sys.path.append('/home/yy28/rsync/structured-sparsity/jax-extend')

import jax.numpy as jnp
import jax
import flax.linen as nn
from jax.nn.initializers import glorot_normal
from flax.linen import softmax
import sparse_attention

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
    batch_size : int
    sequence_length : int
    hidden_dim : int
    head_dim : int
    num_heads : int
    calc_sparse_attention: Any
    dropout : float = 0.1
    mask : bool = False
    downsampling_k : int = 64 ## Default it to 64.
    dtype: Optional[Dtype] = None
    param_dtype: Any = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    use_bias: bool = True
    precision: nn.linear.PrecisionLike = None
    numerical_stabilizer: float = 0.001
    up_train: bool = False
    sparsity_param: int = 128

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
        self.dense_out = DenseGeneral(features=self.hidden_dim,
                           axis=(-2, -1),
                           kernel_init=self.kernel_init,
                           bias_init=self.bias_init,
                           use_bias=self.use_bias,
                           dtype=self.dtype,
                           param_dtype=self.param_dtype,
                           precision=self.precision,
                           name='out')



    def __call__(self, x, step, *, train):
        query, key, value = x

        queries, keys, values = (self.dense_queries(query),
                                 self.dense_keys(key),
                                 self.dense_values(value))

        a_v = self.calc_sparse_attention(queries, keys, values)

        out = self.dense_out(a_v)
        return out

'''
from jax import random

hidden_dim = 10
head_dim = 10
num_heads = 1
batch_size = 1
sequence_length = 10

kernel = sparse_attention.SparseAttention(batch_size=batch_size,
                                                              sequence_length=sequence_length,
                                                              num_heads=num_heads,
                                                              hidden_dim=hidden_dim,
                                                              sparsity_param=2)

mha = MHA(batch_size, sequence_length, hidden_dim, head_dim, num_heads, calc_sparse_attention=kernel)

param_key = random.PRNGKey(42)
dropout_key = random.PRNGKey(43)
qs = random.uniform(random.PRNGKey(44), (batch_size, sequence_length, hidden_dim))
ks = random.uniform(random.PRNGKey(45), (batch_size, sequence_length, hidden_dim))
vs = random.uniform(random.PRNGKey(46), (batch_size, sequence_length, hidden_dim))
params = mha.init({'params': param_key, 'dropout': dropout_key}, [qs, ks, vs], step=0, train=True)
attention_mat = mha.apply(params, [qs, ks, vs], step=0, train=False, rngs={'dropout': dropout_key})
print(attention_mat)
print(attention_mat.shape)
'''