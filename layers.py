# Copyright 2020 The Sabertooth Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layers used in a Transformer."""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from efficient_attention.Linformer.lin_mha import MHA as LinMHA
from efficient_attention.Performer.performer_mha import MHA as PerfMHA
from efficient_attention.SONIC.sonic_lin_perf_mha import MHA as LinPerfMHA
from efficient_attention.SONIC.sonic_lin_rfa_mha import MHA as LinRFAMHA
from efficient_attention.RFA.rfa_mha import MHA as RFAMHA
from efficient_attention.Transformers_are_RNNs.RNNs_mha import MHA as RNNsMHA
from efficient_attention.SONIC.sonic_lin_RNNs_mha import MHA as LinRNNsMHA
from efficient_attention.EVA.eva_mha import MHA as EVAMHA
from efficient_attention.SONIC.sonic_lin_eva_mha import MHA as LinEVAMHA

def gelu(x):
    return jax.nn.gelu(x, approximate=False)


def truncated_normal_initializer(stddev=0.02, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        return jax.random.truncated_normal(key, -2, 2, shape, dtype) * stddev

    return init


class PositionalEncoding(nn.Embed):
    """Learned positional embeddings for the Transformer."""

    # num_embeddings: int
    # features: int
    # dtype: Dtype = jnp.float32
    # embedding_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_embed_init
    # embedding: Array = field(init=False)

    def __call__(self, inputs):
        """Applies PositionalEncoding module."""
        assert inputs.ndim in (
            2,
            3,
        ), f"Number of dimention should be 2 or 3, but it is: {inputs.ndim}"
        length = inputs.shape[1]
        assert length <= self.num_embeddings, (
            f"Sequence is too long for position emdeddings"
            " (length {length}, expected at most {self.num_embeddings})"
        )
        return self.embedding[None, :length, :]


class FeedForward(nn.Module):
    """Feed-forward layer for a Transformer model."""

    d_model: int
    d_ff: int
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    intermediate_activation: Callable[..., Any] = gelu
    kernel_init: Callable[..., Any] = truncated_normal_initializer(0.02)

    def setup(self):
        self.intermediate = nn.Dense(
            self.d_ff,
            kernel_init=self.kernel_init,
            name="intermediate",
            dtype=self.dtype,
        )
        self.output = nn.Dense(
            self.d_model, kernel_init=self.kernel_init, name="output"
        )

    def __call__(self, hidden_states, *, deterministic=False):
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.intermediate_activation(hidden_states)
        hidden_states = self.output(hidden_states)
        return hidden_states


class SelfAttention(nn.SelfAttention):
    """Self-attention, but expecting a different format for mask."""

    @nn.compact
    def __call__(self, hidden_states, switch, mask=None, *, deterministic=False):
        # Attention mask input has mask.shape == (batch_size, kv_length)
        # Flax instead expects mask.shape == (batch_size, 1, 1, kv_length)
        if mask is not None:
            mask = jnp.expand_dims(mask, axis=(-3, -2))
        return super().__call__(hidden_states, mask, deterministic=deterministic)

class FastSelfAttention(nn.Module):
    hidden_dim: int
    head_dim: int
    num_heads: int
    dropout: float
    attention_type: str
    downsampling_k: int = 64
    up_train: bool = False
    use_t5_rpe: bool = False
    overlap_window: bool = False
    window_size: int = 2
    num_landmarks: int = 49
    def setup(self):
        ## We first have the pre-ambulatory initialization.
        # pdb.set_trace()
        if self.attention_type == "PerfMHA":
            self.mha = PerfMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads,
                               dropout=self.dropout, mask=False, up_train=self.up_train)
        elif self.attention_type == "LinMHA":
            self.mha = LinMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads,
                              dropout=self.dropout, mask=False, downsampling_k=self.downsampling_k, up_train=self.up_train)
        elif self.attention_type == "LinPerfMHA":
            self.mha = LinPerfMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads,
                                  dropout=self.dropout, mask=False, downsampling_k=self.downsampling_k, up_train=self.up_train)
        elif self.attention_type == "LinRFAMHA":
            self.mha = LinRFAMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads,
                                 dropout=self.dropout, mask=False, downsampling_k=self.downsampling_k, up_train=self.up_train)
        elif self.attention_type == "RFAMHA":
            self.mha = RFAMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads,
                              dropout=self.dropout, mask=False, up_train=self.up_train)
        elif self.attention_type == "RNNsMHA":
            self.mha = RNNsMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads,
                               dropout=self.dropout, mask=False, up_train=self.up_train)
        elif self.attention_type == "LinRNNsMHA":
            self.mha = LinRNNsMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads,
                                  dropout=self.dropout, mask=False, downsampling_k=self.downsampling_k, up_train=self.up_train)
        elif self.attention_type == "EVAMHA":
            self.mha = EVAMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads,
                              dropout=self.dropout, mask=False, up_train=self.up_train, use_t5_rpe=self.use_t5_rpe,
                              window_size=self.window_size, num_landmarks=self.num_landmarks, overlap_window=self.overlap_window )
        elif self.attention_type == "LinEVAMHA":
            self.mha = LinEVAMHA(hidden_dim=self.hidden_dim, head_dim=self.head_dim, num_heads=self.num_heads,
                                 dropout=self.dropout, mask=False, downsampling_k=self.downsampling_k,
                                 up_train=self.up_train, use_t5_rpe=self.use_t5_rpe, window_size=self.window_size,
                                 num_landmarks=self.num_landmarks, overlap_window=self.overlap_window)
        else:
            raise Exception("Incorrect input of attention_type!")


    @nn.compact
    def __call__(self, hidden_states, switch, layer_num, mask=None, *, deterministic=False):
        # Attention mask input has mask.shape == (batch_size, kv_length)
        # Flax instead expects mask.shape == (batch_size, 1, 1, kv_length)
        if mask is not None:
            mask = jnp.expand_dims(mask, axis=(-3, -2))
        queries, keys, values = hidden_states, hidden_states, hidden_states
        attn = self.mha([queries, keys, values], switch, layer_num, train=not deterministic)
        return attn


class TransformerBlock(nn.Module):
    """Transformer block with normalization after each sub-layer."""

    build_feed_forward: Callable[..., Any]
    build_self_attention: Callable[..., Any]
    dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-12

    def setup(self):
        self.self_attention = self.build_self_attention()
        self.self_attention_dropout = nn.Dropout(rate=self.dropout_rate)
        self.self_attention_layer_norm = nn.LayerNorm(epsilon=self.layer_norm_epsilon)
        self.feed_forward = self.build_feed_forward()
        self.output_dropout = nn.Dropout(rate=self.dropout_rate)
        self.output_layer_norm = nn.LayerNorm(epsilon=self.layer_norm_epsilon)

    def __call__(self, hidden_states, mask, switch, layer_num, *, deterministic=False):
        attention_output = self.self_attention(
            hidden_states, switch, mask, layer_num, deterministic=deterministic
        )
        attention_output = self.self_attention_dropout(
            attention_output, deterministic=deterministic
        )
        hidden_states = self.self_attention_layer_norm(hidden_states + attention_output)
        feed_forward_output = self.feed_forward(
            hidden_states, deterministic=deterministic
        )
        feed_forward_output = self.output_dropout(
            feed_forward_output, deterministic=deterministic
        )
        hidden_states = self.output_layer_norm(hidden_states + feed_forward_output)
        return hidden_states


class OutputProjection(nn.Module):
    """A dense projection layer for computing output logits."""

    n_out: Optional[int] = None
    use_bias: bool = True
    kernel_init: Callable[..., Any] = truncated_normal_initializer(0.02)
    bias_init: Callable[..., Any] = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, kernel: jnp.ndarray = None):
        """Applies OutputProjection module."""
        if kernel is None:
            assert (
                self.n_out is not None
            ), "n_out argument is required when not re-using an embedding matrix"
            kernel = self.param(
                "kernel", self.kernel_init, (self.n_out, inputs.shape[-1])
            )
        y = jnp.matmul(inputs, jnp.transpose(kernel, (1, 0)))
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (y.shape[-1],))
            y = y + bias
        return y
