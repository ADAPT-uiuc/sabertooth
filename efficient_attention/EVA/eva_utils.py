import math
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import flax.linen as nn

import pdb

def prm_projection(
    data,
    projection_matrix,
    normalize: bool = True,
    diagonal: bool = False,
    return_exp: bool = False,
    is_query: bool = False,
    eps: float = 1e-8):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    batch_dims_t: tuple of batch dimensions
    is_query: predicate indicating whether input data corresponds to queries or
        keys
    eps: numerical stabilizer.
    Returns:
    Random features for fast softmax attention.
    """
    # data : [n, b, h, lk, d]
    # proj : [n, b, h, lc, d]
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    # NOTE: scaler with 0.5 could considerably stablizes training.
    # now test norm also uses scaled data: that is, multiply by data.shape[-1] ** -1.
    # normalized_data = (data.shape[-1] ** -0.5) * data
    # data_dash = torch.einsum('...nd,...md->...nm',
    #                         projection_matrix,
    #                         normalized_data,
    #                         ) # [n, b, h, c, lq]
    # norm = torch.sum(normalized_data ** 2, dim=-1).unsqueeze(-2) / 2.0# [n, b, h, 1, lk]
    data_normalizer = (data.shape[-1] ** -0.5)
    if diagonal:
        data_dash = jnp.einsum('...nd,...nd->...n',
                            projection_matrix,
                            (data_normalizer * data),
                            ) # [n, b, h, lq, lk]
        norm = data_normalizer * jnp.sum(data ** 2, dim=-1) / 2.0# [n, b, h, 1, lk]
    else:
        data_dash = jnp.einsum('...nd,...md->...nm',
                                projection_matrix,
                                (data_normalizer * data),
                                ) # [n, b, h, lq, lk]
        norm = data_normalizer * jnp.expand_dims(jnp.sum(data ** 2, axis=-1), axis=-2) / 2.0# [n, b, h, 1, lk]
    if normalize:
        proj_data = nn.softmax(data_dash - norm, axis=-1)  # [n, b, h, l_c, l_k]
    elif return_exp:
        if is_query:
            proj_data = jnp.exp(
                data_dash - norm - jnp.amax(data_dash, axis=-2, keepdim=True)) + eps
        else:
            proj_data = jnp.exp(
                data_dash - norm - jnp.amax(data_dash, axis=(-1, -2, -3), keepdim=True)) + eps
    else:
        proj_data = data_dash - norm
    return proj_data

def truncated_normal(key, shape, mean=0.0, std=0.2, lower=-2.0, upper=2.0):
    """Truncated normal initializer."""
    samples = jax.random.normal(key, shape) * std * math.sqrt(2.) + mean
    return jnp.clip(samples, lower, upper)


def pad_to_multiple(tensor, multiple, dim=-2, value=0, create_mask=False):
    assert dim < 0 # only accept ``dim'' index in a reverse manner
    seqlen = int(tensor.shape[dim])
    m = seqlen / multiple
    if m.is_integer():
        if create_mask:
            return tensor, jnp.zeros(shape=(tensor.shape[0], tensor.shape[-2]), dtype=jnp.bool_)
        else:
            return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    padded_res = jnp.pad(tensor, (*pad_offset, 0, remainder), constant_values=value)
    if create_mask:
        # assume dim 0 is the batch size
        padding_mask = jnp.zeros(shape=(padded_res.shape[0], padded_res.shape[-2]), dtype=jnp.bool_)
        padding_mask[:, -remainder:] = True
        return padded_res, padding_mask
    else:
        return padded_res

def nonoverlap_window_1d_partition(x, window_size):
    assert x.shape[2] % window_size == 0
    g = x.shape[2] // window_size
    return jnp.reshape(x, (x.shape[0], x.shape[1], g, window_size, x.shape[-1]))


@partial(jax.jit, static_argnums=(1,))
def moving_window(matrix, size):
    window_size, ext_size = size
    matrix_width = matrix.shape[1]
    matrix_height = matrix.shape[0]

    window_width = matrix_width
    window_height = window_size + 2 * ext_size

    startsx = jnp.arange(matrix_width - window_width + 1)
    startsy = jnp.arange(matrix_height - window_height + 1, step=window_size)
    starts_xy = jnp.dstack(jnp.meshgrid(startsx, startsy)).reshape(-1, 2) # cartesian product => [[x,y], [x,y], ...]

    return jax.vmap(lambda start: jax.lax.dynamic_slice(matrix, (start[1], start[0]), (window_height, window_width)))(starts_xy)


def window_1d_partition(x, window_size, ext_window_size=0, pad_val=0):
    pdb.set_trace()
    b, h, n, d = x.shape
    n_groups = n // window_size
    if ext_window_size > 0:
        ext_len = ext_window_size
        x = jnp.pad(x, ((0, 0), (0, 0), (ext_len, ext_len), (0, 0)), constant_values=pad_val)
        '''
        out_shape = (b, h, n_groups, 2 * ext_len + window_size, d)
        strides = x.strides
        out_stride = (strides[0], strides[1], window_size * strides[2], strides[2], strides[3])
        return torch.as_strided(x, size=out_shape, stride=out_stride)
        '''
        s = x.shape
        matrix = x.reshape((s[0] * s[1], s[2], s[3]))
        matrix = jax.vmap(moving_window, in_axes=(0, None))(matrix, (window_size, ext_window_size))
        w = jnp.reshape(matrix, (s[0], s[1], matrix.shape[1], matrix.shape[2], matrix.shape[3]))
        return w
    else:
        return nonoverlap_window_1d_partition(x, window_size)


def window_1d_merge(x):
    gw = x.shape[-2] * x.shape[-3]
    return jnp.reshape(x, (x.shape[0], x.shape[1], gw, x.shape[-1]))


class T5RelativePositionBias(nn.Module):
    scale: float
    num_heads: int
    causal: bool = False
    num_buckets: int = 32
    max_distance: int = 128

    def setup(self):
        self.relative_attention_bias = nn.Embed(num_embeddings=self.num_buckets, features=self.num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal=True,
        num_buckets=32,
        max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).astype(jnp.int64) * num_buckets
            n = jnp.abs(n)
        else:
            n = jnp.max(n, jnp.zeros_like(n))
            # used np instead of jnp

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            jnp.log(n.astype(float) / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).astype(jnp.int64)
        val_if_large = jnp.minimum(val_if_large, np.full_like(val_if_large, num_buckets - 1))

        ret += jnp.where(is_small, n, val_if_large)
        return ret

    def __call__(self, x):
        i, j = x.shape[-2:]
        q_pos = jnp.arange(i, dtype=jnp.int64)
        k_pos = jnp.arange(j, dtype=jnp.int64)
        # rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rel_pos = jnp.expand_dims(k_pos, axis=0) - jnp.expand_dims(q_pos, axis=1)


        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets, max_distance=self.max_distance)
        bias = jnp.expand_dims(jnp.expand_dims(self.relative_attention_bias(rp_bucket).transpose(2, 0, 1), axis=0), axis=2)
        return bias * self.scale


