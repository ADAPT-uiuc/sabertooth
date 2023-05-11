import sys
import os

import jax.numpy as jnp
import numpy as np

import flax.linen as nn
import jax.random
from flax.linen.linear import DenseGeneral, default_kernel_init

import functools
from typing import (Any, Callable, Optional, Tuple)

from efficient_attention.EVA.eva_utils import T5RelativePositionBias, truncated_normal,\
    pad_to_multiple, window_1d_partition, prm_projection, window_1d_merge

sys.path.append(os.getcwd())

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

import pdb ## For debugging purposes only.


class MHA(nn.Module):
    hidden_dim : int
    head_dim : int
    num_heads : int
    dropout : float
    mask : bool
    downsampling_k : int = 64 ## Default it to 64.
    use_rpe: bool = False
    window_size: int = 2
    attn_2d: bool = False
    overlap_window: bool = False
    dtype: Optional[Dtype] = None
    param_dtype: Any = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    use_bias: bool = True
    precision: nn.linear.PrecisionLike = None
    numerical_stabilizer: float = 0.001
    up_train: bool = False
    adaptive_proj: str = 'default'
    num_landmarks: int = 49
    use_t5_rpe: bool = False


    def setup(self):
        downsampling_shape = (self.downsampling_k, 1024)
        mean = 0.0
        sd = float(1)/float(self.downsampling_k)

        self.key_downsampling_mat = self.param('key_downsample_mat', lambda rng, shape, mean, sd: mean + sd * jax.random.normal(rng, shape=shape), downsampling_shape, mean, sd)
        self.value_downsampling_mat = self.param('value_downsample_mat', lambda rng, shape, mean, sd: mean + sd * jax.random.normal(rng, shape=shape), downsampling_shape, mean, sd)

        self.dense_queries = DenseGeneral(axis=-1, dtype=self.dtype, param_dtype=self.param_dtype,
                                          features=(self.num_heads, self.head_dim), kernel_init=self.kernel_init,
                                          bias_init=self.bias_init, use_bias=self.use_bias, precision=self.precision,
                                          name='query')
        self.dense_keys = DenseGeneral(axis=-1, dtype=self.dtype, param_dtype=self.param_dtype,
                                       features=(self.num_heads, self.head_dim), kernel_init=self.kernel_init,
                                       bias_init=self.bias_init, use_bias=self.use_bias, precision=self.precision,
                                       name='key')
        self.dense_values = DenseGeneral(axis=-1, dtype=self.dtype, param_dtype=self.param_dtype,
                                         features=(self.num_heads, self.head_dim), kernel_init=self.kernel_init,
                                         bias_init=self.bias_init, use_bias=self.use_bias, precision=self.precision,
                                         name='value')
        self.dense_out = DenseGeneral(features=self.hidden_dim,
                                      axis=(-2, -1),
                                      kernel_init=self.kernel_init,
                                      bias_init=self.bias_init,
                                      use_bias=self.use_bias,
                                      dtype=self.dtype,
                                      param_dtype=self.param_dtype,
                                      precision=self.precision,
                                      name='out')

        self.scale = self.head_dim ** -0.5
        self.dropout_layer = nn.Dropout(self.dropout)

        if self.overlap_window:
            self.ext_size = max(1, self.window_size // 2)
        else:
            self.ext_size = 0

        if self.use_rpe:
            if self.attn_2d:
                # TODO
                raise NotImplementedError
            else:
                self.local_relative_position_bias_table = self.param(
                    'local_relative_position_bias_table',
                    truncated_normal,
                    (self.num_heads, self.window_size, self.window_size + self.ext_size * 2),
                )

        if self.adaptive_proj in ['default']:
            self.adaptive_mu_q = nn.Sequential([nn.Dense(self.head_dim, self.head_dim),
                                               nn.LayerNorm(self.head_dim)])
            self.adaptive_mu_k = nn.Sequential([nn.Dense(self.head_dim, self.head_dim),
                                               nn.LayerNorm(self.head_dim)])
        else:
            raise NotImplementedError

        if self.use_t5_rpe:
            self.rel_pos_bias = T5RelativePositionBias(self.scale,
                                                       num_heads=self.num_heads,
                                                       causal=False,
                                                       num_buckets=max(min(int((self.window_size + self.ext_size) / 2), 64), 16),
                                                       max_distance=self.window_size + self.ext_size)

        self.random_rng = jax.random.PRNGKey(0)

    def window_partition(self, x, shape, ext_window_size, pad_val=0, window_size=None):
        if window_size is None:
            window_size = self.window_size
        if self.attn_2d:
            raise NotImplementedError
        else:
            return window_1d_partition(
                x,
                window_size=window_size,
                ext_window_size=ext_window_size,
                pad_val=pad_val
                )

    def window_merge(self, x, shape, window_size=None):
        if window_size is None:
            window_size = self.window_size
        if self.attn_2d:
            '''
            assert isinstance(shape, (list, tuple))
            output = window_2d_merge(
                x,
                window_size=window_size, 
                hw_tuple=shape
                )
            return rearrange(output, '... H W d ->... (H W) d')
            '''
            raise NotImplementedError
        else:
            return window_1d_merge(x)

    def _process_input(self, x, key_padding_mask=None):
        B, N, C = x.shape
        if self.attn_2d:
            raise NotImplementedError
        else:
            if self.window_size > 0:
                if key_padding_mask is None:
                    x, key_padding_mask = pad_to_multiple(x, self.window_size, dim=-2, create_mask=True)
                else:
                    x = pad_to_multiple(x, self.window_size, dim=-2)
                    key_padding_mask = pad_to_multiple(key_padding_mask, self.window_size, dim=-1, value=True)
                N = x.shape[-2]
        return x, key_padding_mask, N


    def __call__(self, x, switch: bool, *, train, key_padding_mask=None):
        ## Jax complains about passing in multiple arguments.
        ## So we do the hack of concatenating the queries, keys and values into a list and unpacking it.
        mask_val = -5e4
        query, key, value = x
        batch_size, sequence_length, hidden_dims = query.shape
        # pdb.set_trace()

        assert all(len(i.shape) == 3 for i in x), "Incorrect size of input, should be [batch, seq length, hidden dimension]"
        key = jnp.einsum('ks, bsd -> bkd', self.key_downsampling_mat, key)
        value = jnp.einsum('ks, bsd -> bkd', self.value_downsampling_mat, value)

        if self.up_train and not switch:
            # Uptrain with Linformer attention
            queries, keys, values = (self.dense_queries(query),
                                     self.dense_keys(key),
                                     self.dense_values(value))

            q_ks = jnp.einsum('bqnh, bknh -> bnqk', queries, keys)

            q_ks /= jnp.sqrt(jnp.array(self.head_dim).astype(np.float32))

            attn_mat = nn.softmax(q_ks)

            attn_mat = self.dropout_layer(attn_mat, deterministic=not train)

            a_v = jnp.einsum('bhqk, bkhd -> bqhd', attn_mat, values)

            out = self.dense_out(a_v)
            return out

        query, key_padding_mask, seq_shape = self._process_input(query, key_padding_mask)
        key, key_padding_mask, seq_shape = self._process_input(key, key_padding_mask)
        values, key_padding_mask, seq_shape = self._process_input(value, key_padding_mask)

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        queries, keys, values = (self.dense_queries(query).transpose(0, 2, 1, 3),
                                 self.dense_keys(key).transpose(0, 2, 1, 3),
                                 self.dense_values(value).transpose(0, 2, 1, 3))
        #queries, keys, values = (query.reshape((batch_size, sequence_length, num_heads, head_dim)).transpose(0, 2, 1, 3),
        #                         key.reshape((batch_size, sequence_length, num_heads, head_dim)).transpose(0, 2, 1, 3),
        #                         value.reshape((batch_size, sequence_length, num_heads, head_dim)).transpose(0, 2, 1, 3))


        if key_padding_mask is None:
            key_padding_mask = jnp.zeros((batch_size, sequence_length), dtype=keys.dtype)
        key_padding_mask = jnp.expand_dims(jnp.expand_dims(key_padding_mask, axis=1), axis=-1).astype(bool) # [b, 1, n, 1]

        w_q = self.window_partition(queries, sequence_length, window_size=self.window_size * (sequence_length // self.downsampling_k), ext_window_size=0)
        w_k = self.window_partition(keys, self.downsampling_k, ext_window_size=self.ext_size)
        w_v = self.window_partition(values, self.downsampling_k, ext_window_size=self.ext_size)

        if self.attn_2d:
            raise NotImplementedError
        else:
            assert sequence_length % self.num_landmarks == 0 and self.downsampling_k % self.num_landmarks == 0
            rf_win_size_q = int(sequence_length // self.num_landmarks)
            rf_win_size_kv = int(self.downsampling_k // self.num_landmarks)

        rf_w_q = self.window_partition(queries, sequence_length, window_size=rf_win_size_q, ext_window_size=self.ext_size)
        rf_w_k = self.window_partition(keys, sequence_length, window_size=rf_win_size_kv, ext_window_size=self.ext_size)
        rf_w_v = self.window_partition(values, sequence_length, window_size=rf_win_size_kv, ext_window_size=self.ext_size)

        # input mask is not used in Linformer attention
        '''
        rf_w_mask = self.window_partition(
            key_padding_mask,
            sequence_length,
            window_size=rf_win_size,
            ext_window_size=self.ext_size,
            pad_val=1
        ).astype(bool)

        rf_w_q = jnp.where(rf_w_mask, 0., rf_w_q)
        rf_w_k = jnp.where(rf_w_mask, 0., rf_w_k)
        rf_w_v = jnp.where(rf_w_mask, 0., rf_w_v)
        '''

        if self.adaptive_proj in ['default', 'no-ln']:
            rf_q_bar = self.adaptive_mu_q(rf_w_q.mean(axis=-2))
            rf_k_bar = self.adaptive_mu_k(rf_w_k.mean(axis=-2))
            # [b, h, c, d]
            mu = 0.5 * (rf_q_bar + rf_k_bar)
        else:
            raise NotImplementedError

        if train:
            random_rng, _ = jax.random.split(self.random_rng)
            weights = mu + jax.random.normal(random_rng, shape=mu.shape) # TODO:check normal
        else:
            weights = mu

        log_proj_w_k = prm_projection(rf_w_k, jnp.expand_dims(weights, axis=-2), normalize=False).squeeze(-2)
        # log_proj_w_k = jnp.where(jnp.squeeze(rf_w_mask, axis=-1), mask_val, log_proj_w_k)

        beta = jnp.einsum('...cj,...cjd->...cd', nn.softmax(log_proj_w_k, axis=-1), rf_w_v)

        rfa_chunk = jnp.einsum('...wid,...cd->...wic', w_q, self.scale * rf_k_bar)
        num_rfa_chunks = rfa_chunk.shape[-1]

        '''
        local_dots_mask = self.window_partition(
            key_padding_mask,
            sequence_length,
            ext_window_size=self.ext_size,
            pad_val=1
        ).astype(bool).transpose((0, 1, 2, -1, -2))
        '''

        log_qk_local_dot = jnp.einsum('bhwie,bhwje->bhwij', w_q, w_k) * self.scale # [b, h, w, i, j]

        if self.use_t5_rpe:
            # here the t5-rpe-bias has already been scaled by \sqrt{d}
            log_qk_local_dot = log_qk_local_dot + self.rel_pos_bias(log_qk_local_dot)
        if self.use_rpe:
            raise NotImplementedError
            # log_qk_local_dot = self.add_rel_pos_bias(log_qk_local_dot)

        # log_qk_local_dot = log_qk_local_dot.masked_fill(local_dots_mask, mask_val)
        # log_qk_local_dot = jnp.where(local_dots_mask, mask_val, log_qk_local_dot)

        local_len = log_qk_local_dot.shape[-1]

        attn = nn.softmax(jnp.concatenate([log_qk_local_dot, rfa_chunk], axis=-1), axis=-1)
        local_attn, ra_attn = jnp.split(attn, [local_len], axis=-1)

        output_local = jnp.einsum('bhwij,bhwjd->bhwid', local_attn, w_v)
        output_snis = jnp.einsum('bhwic,bhcd->bhwid', ra_attn, beta)

        ######################## Combine them together ############################
        output = self.window_merge(output_snis + output_local, sequence_length) # [b, h, n, d]
        x = jnp.transpose(output, (0, 2, 1, 3))

        out = self.dense_out(x)
        return out

"""
from jax import random

hidden_dim = 32
head_dim = 8
num_heads = 4
dropout = 0.1
mask = False
batch_size = 2
sequence_length = 128
mha = MHA(
    hidden_dim=hidden_dim, head_dim=head_dim, num_heads=num_heads, dropout=dropout, mask=mask,
    use_rpe=False, window_size=4, overlap_window=True, num_landmarks=4, use_t5_rpe=True,
    up_train=True)

# x = random.uniform(random.PRNGKey(44), (batch_size, sequence_length, hidden_dim))
x = jnp.ones((batch_size, sequence_length, hidden_dim))
param_key = random.PRNGKey(42)
dropout_key = random.PRNGKey(43)
params = mha.init({'params': param_key, 'dropout': dropout_key}, [x, x, x], switch=True, train=True)
attention_mat = mha.apply(params, [x, x, x], switch=False, train=True, rngs={'dropout': dropout_key})
attention_mat = mha.apply(params, [x, x, x], switch=True, train=True, rngs={'dropout': dropout_key})
print(attention_mat)
"""

