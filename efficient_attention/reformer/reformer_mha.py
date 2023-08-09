from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
from flax.linen.linear import DenseGeneral, default_kernel_init
from typing import (Any, Callable, Optional, Tuple)
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


def look_one_back(x):
  """Looks back to previous chunk.

  Args:
    x: input tensor of shape [num_chunks, div_len, dim]
  Returns:
    output tensor of shape [num_chunks, div_len * 2, dim]
  """
  xlb = jnp.concatenate([x[-1:, ...], x[:-1, ...]], axis=0)
  return jnp.concatenate([x, xlb], axis=1)


'''
def permute_via_gather(val, permutation, inverse_permutation, axis=0):
  """Permutation helper for LSH attention."""
  # It is *not* safe to use jax.custom_vjp here. The most likely cause is that
  # it can't close over values: https://github.com/google/jax/issues/2676
  # The error only occurs in some configurations (e.g. use_python_loop = True,
  # num_parallel_heads = 1) but not others.
  permutation = jax.lax.stop_gradient(permutation)
  inverse_permutation = jax.lax.stop_gradient(inverse_permutation)
  def permute_impl(val):
    return jnp.take(val, permutation, axis=axis)
  def permute_vjp(val):
    permuted = permute_impl(jax.lax.stop_gradient(val))
    def vjpfun(permuted_grad):
      # JAX autodiff would synthesize a scatter operation because it doesn't
      # know that the indices are a permutatation. However on TPU, gathers are
      # faster than scatters (at least in the regime the LSH attention uses).
      return (jnp.take(permuted_grad, inverse_permutation, axis=axis),)
    return permuted, vjpfun
  permute = jax.custom_transforms(permute_impl)
  jax.defvjp_all(permute, permute_vjp)
  return permute(val)
'''

def permute_via_gather(val, permutation, inverse_permutation, axis=0):
    """Permutation helper for LSH attention."""
    permutation = jax.lax.stop_gradient(permutation)
    inverse_permutation = jax.lax.stop_gradient(inverse_permutation)

    @jax.custom_vjp
    def permute(val):
      return jnp.take(val, permutation, axis=axis)

    def permute_fwd(val):
      permuted = permute(jax.lax.stop_gradient(val))
      return permuted, (val, inverse_permutation, axis)

    def permute_bwd(res, permuted_grad):
      val, inverse_permutation, axis = res
      vjp = jnp.take(permuted_grad, inverse_permutation, axis=axis)
      return (vjp,)

    permute.defvjp(permute_fwd, permute_bwd)

    return permute(val)

'''
def permute_via_sort(val, keys, inverse_keys, axis=0):
  """Permutation helper for LSH attention."""
  # It is *not* safe to use jax.custom_vjp here (see permute_via_gather).
  keys = jax.lax.stop_gradient(keys)
  inverse_keys = jax.lax.stop_gradient(inverse_keys)
  def permute_impl(val):
    # On TPU, sorting scalars by key is faster than a gather.
    _, permuted = jax.lax.sort_key_val(keys, val, dimension=axis)
    return permuted
  def permute_vjp(val):
    permuted = permute_impl(jax.lax.stop_gradient(val))
    def vjpfun(permuted_grad):
      _, val_grad = jax.lax.sort_key_val(
          inverse_keys, permuted_grad, dimension=axis)
      return (val_grad,)
    return permuted, vjpfun
  permute = jax.custom_transforms(permute_impl)
  jax.defvjp_all(permute, permute_vjp)
  return permute(val)
'''


def permute_via_sort(val, keys, inverse_keys, axis=0):
  """Permutation helper for LSH attention."""
  keys = jax.lax.stop_gradient(keys)
  inverse_keys = jax.lax.stop_gradient(inverse_keys)

  @jax.custom_vjp
  def permute(val):
    # On TPU, sorting scalars by key is faster than a gather.
    _, permuted = jax.lax.sort_key_val(keys, val, dimension=axis)
    return permuted

  def permute_fwd(val):
    permuted = permute(jax.lax.stop_gradient(val))
    return permuted, (val, inverse_keys, axis)

  def permute_bwd(res, permuted_grad):
    val, inverse_keys, axis = res
    _, val_grad = jax.lax.sort_key_val(
      inverse_keys, permuted_grad, dimension=axis)
    return (val_grad,)

  permute.defvjp(permute_fwd, permute_bwd)

  return permute(val)



def hash_vectors(vecs, rng, num_buckets, num_hashes):
  """Performs batched hashing.

  Args:
    vecs: input of [length, dim].
    rng: rng object.
    num_buckets: integer, number of buckets.
    num_hashes: integer, number of hashes.
  Returns:
    output of shape [batch_size, length]
  """

  # batch_size = vecs.shape[0]

  assert num_buckets % 2 == 0

  rot_size = num_buckets

  rotations_shape = (vecs.shape[-1], num_hashes, rot_size // 2)

  rng = jax.lax.stop_gradient(jax.lax.tie_in(vecs, rng))
  random_rotations = jax.random.normal(rng, rotations_shape).astype(jnp.float32)

  rotated_vecs = jnp.einsum('tf,fhi->hti', vecs, random_rotations)

  rotated_vecs = jnp.concatenate([rotated_vecs, -rotated_vecs], axis=-1)
  buckets = jnp.argmax(rotated_vecs, axis=-1)
  offsets = jax.lax.tie_in(buckets, jnp.arange(num_hashes))
  offsets = jnp.reshape(offsets * num_buckets, (-1, 1))
  buckets = jnp.reshape(buckets + offsets, (-1,))

  return buckets


def lsh_attention_single_batch(query, value, n_buckets, n_hashes,
                               causal_mask=True):
  """LSH attention for single batch."""
  del causal_mask
  attn = jax.vmap(lsh_attention_single_head, in_axes=(1, 1, None, None))
  out = attn(query, value, n_buckets, n_hashes)
  return out


def length_normalized(x, epsilon=1e-6):
  variance = jnp.mean(x**2, axis=-1, keepdims=True)
  norm_inputs = x / jnp.sqrt(variance + epsilon)
  return norm_inputs


def lsh_attention_single_head(query, value, n_buckets, n_hashes,
                              causal_mask=True,
                              length_norm=False):
  """Applies LSH attention on a single head and a single batch.

  Args:
    query: query tensor of shape [qlength, dims].
    value: value tensor of shape [vlength, dims].
    n_buckets: integer, number of buckets.
    n_hashes: integer, number of hashes.
    causal_mask: boolean, to use causal mask or not.
    length_norm: boolean, to normalize k or not.
  Returns:
    output tensor of shape [qlength, dims]
  """

  qdim, vdim = query.shape[-1], value.shape[-1]
  chunk_size = n_hashes * n_buckets

  seqlen = query.shape[0]

  rng = jax.random.PRNGKey(0)

  buckets = hash_vectors(
      query, rng, num_buckets=n_buckets, num_hashes=n_hashes)
  # buckets should be (seq_len)
  assert buckets.shape[-1] == n_hashes * seqlen

  total_hashes = n_hashes

  # create sort and unsort
  ticker = jax.lax.tie_in(query, jnp.arange(n_hashes * seqlen))
  buckets_and_t = seqlen * buckets + (ticker % seqlen)
  buckets_and_t = jax.lax.stop_gradient(buckets_and_t)
  # ticker = jnp.tile(jnp.reshape(ticker, [1, -1]), [batch_size, 1])
  sbuckets_and_t, sticker = jax.lax.sort_key_val(
      buckets_and_t, ticker, dimension=-1)
  _, undo_sort = jax.lax.sort_key_val(sticker, ticker, dimension=-1)
  sbuckets_and_t = jax.lax.stop_gradient(sbuckets_and_t)
  sticker = jax.lax.stop_gradient(sticker)
  undo_sort = jax.lax.stop_gradient(undo_sort)

  st = (sticker % seqlen)

  sqk = jnp.take(query, st, axis=0)
  sv = jnp.take(value, st, axis=0)

  bkv_t = jnp.reshape(st, (chunk_size, -1))
  bqk = jnp.reshape(sqk, (chunk_size, -1, qdim))
  bv = jnp.reshape(sv, (chunk_size, -1, vdim))
  bq = bqk
  bk = bqk

  if length_norm:
    bk = length_normalized(bk)

  # get previous chunks
  bk = look_one_back(bk)
  bv = look_one_back(bv)
  bkv_t = look_one_back(bkv_t)

  # compute dot product attention
  dots = jnp.einsum('hie,hje->hij', bq, bk) * (qdim ** 0.5)

  if causal_mask:
    # apply causal mask
    # TODO(yitay): This is not working yet
    # We don't need causal reformer for any task YET.
    pass

  dots_logsumexp = logsumexp(dots, axis=-1, keepdims=True)
  slogits = jnp.reshape(dots_logsumexp, [-1])
  dots = jnp.exp(dots - dots_logsumexp)

  x = jnp.matmul(dots, bv)
  x = jnp.reshape(x, [-1, qdim])

  # Unsort
  o = permute_via_gather(x, undo_sort, sticker, axis=0)
  logits = permute_via_sort(slogits, sticker, undo_sort, axis=0)
  logits = jnp.reshape(logits, [total_hashes, seqlen, 1])
  probs = jnp.exp(logits - logsumexp(logits, axis=0, keepdims=True))
  o = jnp.reshape(o, [n_hashes, seqlen, qdim])
  out = jnp.sum(o * probs, axis=0)
  out = jnp.reshape(out, [seqlen, qdim])

  return out


class MHA(nn.Module):
  hidden_dim: int
  head_dim: int
  num_heads: int
  dropout: float
  mask: bool
  downsampling_k: int = 64  ## Default it to 64.
  dtype: Any = jnp.float32
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  use_bias: bool = True
  precision: nn.linear.PrecisionLike = None
  chunk_len: int = 128
  n_chunks_before: int = 1
  n_hashes: int = 2
  n_buckets: int = 64
  def setup(self):
    self.dense_query = nn.DenseGeneral(
        axis=-1,
        features=(self.num_heads, self.head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision, dtype=self.dtype, name='query')

    self.dense_key = nn.DenseGeneral(
      axis=-1,
      features=(self.num_heads, self.head_dim),
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision, dtype=self.dtype, name='key')

    self.dense_value = nn.DenseGeneral(
      axis=-1,
      features=(self.num_heads, self.head_dim),
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision, dtype=self.dtype, name='value')

    self.dense_out = DenseGeneral(features=self.hidden_dim,
                                  axis=(-2, -1),
                                  kernel_init=self.kernel_init,
                                  bias_init=self.bias_init,
                                  use_bias=self.use_bias,
                                  dtype=self.dtype,
                                  param_dtype=self.dtype,
                                  precision=self.precision,
                                  name='out')

  def __call__(self, x, step, *, train=True):

    assert not self.mask, 'not implemented'
    inputs_q, inputs_k, inputs_v = x
    assert inputs_q.ndim == 3

    qlength = inputs_q.shape[1]
    batch_size = inputs_q.shape[0]

    # chunk_size = n_hashes * n_buckets

    extra_len = self.chunk_len - (qlength % self.chunk_len)
    pad_width = ((0, 0), (0, extra_len), (0, 0))

    inputs_q = jnp.pad(inputs_q, pad_width)
    inputs_k = jnp.pad(inputs_k, pad_width)
    inputs_v = jnp.pad(inputs_v, pad_width)

    assert self.hidden_dim % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')

    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, dims..., n_heads, n_features_per_head]
    query, key, value = (self.dense_query(inputs_q),
                    self.dense_key(inputs_k), self.dense_value(inputs_v))

    attn = jax.vmap(lsh_attention_single_batch, in_axes=(0, 0, None, None))
    out = attn(query, value, self.n_buckets, self.n_hashes)

    out = jnp.transpose(out, (0, 2, 1, 3))
    out = out[:, :qlength, :, :]
    out = self.dense_out(out)
    return out



