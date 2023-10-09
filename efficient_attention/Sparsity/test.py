import sys
sys.path.append('/home/yy28/rsync/structured-sparsity/jax-extend')
import numpy as np
import sparse_attention
print(sparse_attention.__file__)


b = 1
s = 10
n = 1
h = 10
p = 2

x = np.ones(b * s * n * h)
x = x.reshape((b, n, s, h))
x = x.astype(np.float32)

# out = sparse_attention.window_attention(x, x, 2)

#print(out)

calc_sparse_attention = sparse_attention.SparseAttention(batch_size=b, sequence_length=s, num_heads=n, hidden_dim=h, sparsity_param=p)
print(calc_sparse_attention(x, x, x))