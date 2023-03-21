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

import pdb ## For debugging purposes only.

class MHA(nn.Module):
    hidden_dim : int
    head_dim : int
    num_heads : int
    dropout : float
    mask : bool

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

        ## initialize the dropout layer.
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, x, *, train): 
        ## Jax complains about passing in multiple arguments.
        ## So we do the hack of concatenating the queries, keys and values into a list and unpacking it.
        query, key, value = x

        ## First, we map the queries keys and values.
        queries = jnp.einsum('bsd, dnh -> bsnh', query, self.query_kernel)
        keys = jnp.einsum('bsd, dnh -> bsnh', key, self.key_kernel)
        values = jnp.einsum('bsd, dnh -> bsnh', value, self.value_kernel)

        ## Then we compute the product in between the queries and the keys.
        q_ks = jnp.einsum('bqnh, bknh -> bnqk', queries, keys) 

        ## Then we scale it with the head hidden dimension.
        q_ks /= jnp.sqrt(jnp.array(self.head_dim).astype(np.float32))

        if self.mask:
            trilled_mask = jnp.ones((queries.shape[0], queries.shape[2], queries.shape[1], keys.shape[1])).astype(bool) ## This is of size: [batch_size, num_heads, query_seq_length, key_seq_length]
            trilled_mask = jnp.tril(trilled_mask)
            q_ks = jnp.where(trilled_mask == False, -9e15, q_ks)

        ## Then we take the softmax
        attn_mat = softmax(q_ks)

        ## Then we drop out the attention matrix. 
        attn_mat = self.dropout_layer(attn_mat, deterministic=not train)

        ## Then we right multiply by the values and return the result.
        a_v =  jnp.einsum('bhqk, bkhd -> bqhd', attn_mat, values)

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
mask = True 
mha = MHA(hidden_dim, head_dim, num_heads, dropout, mask)

batch_size = 2
sequence_length = 4
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