import math
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    def setup(self):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input array.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        self.weight = self.param("weight", lambda *args: jnp.ones(self.dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input array.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The normalized array.

        """
        return x * jax.lax.rsqrt(jnp.power(x, 2).mean(axis=-1, keepdims=True) + self.eps)

    def __call__(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (jax.Array): The input array.

        Returns:
            jax.Array: The output array after applying RMSNorm.

        """
        output = self._norm(x.astype("float32")).astype(x.dtype)
        return output * self.weight


def polar(r, theta):
    return r * jnp.exp(1j * theta)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency array for complex exponentials (cis) with given dimensions.

    This function calculates a frequency array with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned array contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency array.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        jax.Array: Precomputed frequency array with complex exponentials.
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype("float32") / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs).astype("float32")
    freqs_cis = polar(jnp.ones(freqs.shape, dtype=freqs.dtype), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: jax.Array, x: jax.Array):
    """
    Reshape frequency array for broadcasting it with another array.

    This function reshapes the frequency array to have the same shape as the target array 'x'
    for the purpose of broadcasting the frequency array during element-wise operations.

    Args:
        freqs_cis (jax.Array): Frequency array to be reshaped.
        x (jax.Array): Target array for broadcasting compatibility.

    Returns:
        jax.Array: Reshaped frequency array.

    Raises:
        AssertionError: If the frequency array doesn't match the expected shape.
        AssertionError: If the target array 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(*shape)


def view_as_complex(x: jax.Array):
    return jax.lax.complex(x[..., 0], x[..., 1])

def view_as_real(cx: jax.Array):
    return jnp.stack([jnp.real(cx), jnp.imag(cx)], axis=-1)


def apply_rotary_emb(
    xq: jax.Array,
    xk: jax.Array,
    freqs_cis: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Apply rotary embeddings to input arrays using the given frequency array.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' arrays using the provided
    frequency array 'freqs_cis'. The input arrays are reshaped as complex numbers, and the frequency array
    is reshaped for broadcasting compatibility. The resulting arrays contain rotary embeddings and are
    returned as real arrays.

    Args:
        xq (jax.Array): Query array to apply rotary embeddings.
        xk (jax.Array): Key array to apply rotary embeddings.
        freqs_cis (jax.Array): Precomputed frequency array for complex exponentials.

    Returns:
        Tuple[jax.Array, jax.Array]: Tuple of modified query array and key array with rotary embeddings.
    """
    xq_ = view_as_complex(xq.astype("float32").reshape(*xq.shape[:-1], -1, 2))
    xk_ = view_as_complex(xk.astype("float32").reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = view_as_real(xq_ * freqs_cis)
    xq_out = xq_out.reshape(*xq_out.shape[:3], -1)
    xk_out = view_as_real(xk_ * freqs_cis)
    xk_out = xk_out.reshape(*xk_out.shape[:3], -1)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


def main():
    import torch
    import numpy as np

    batch_size, dim = (3, 4)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (batch_size, dim))
    # xt = torch.from_numpy(np.asarray(x))
    self = RMSNorm(dim)
    variables = self.init(rng, x)
    self = self.bind(variables)
    out = self.apply(variables, x)
    # TODO: check with PyTorch



###############################################################################

def to_pytorch(x):
    import torch
    import numpy as np
    return torch.from_numpy(np.asarray(x))

def to_jax(pt_x):
    return jnp.array(pt_x.detach().numpy())


def test_with_pytorch(module, pt_module, rng, *args):
    # jax
    variables = module.init(rng, *args)
    out = module.apply(variables, *args)
    # pytorch
    pt_args = map(to_pytorch, args)
    pt_out = pt_module(*pt_args)
    assert jnp.all(out == to_jax(pt_out))


def test_rmsnorm():
    from gendo.llama.model_pt import RMSNorm as PtRMSNorm
    batch_size, dim = (3, 4)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (batch_size, dim))
    test_with_pytorch(RMSNorm(dim), PtRMSNorm(dim), rng, x)


def test_precompute_freqs_cis():
    from gendo.llama.model_pt import precompute_freqs_cis as pt_precompute_freqs_cis
    res = precompute_freqs_cis(32, 8)
    pt_res = pt_precompute_freqs_cis(32, 8)
    assert jnp.all(res == to_jax(pt_res))


def test_complex_conversions():
    import torch
    rng = jax.random.PRNGKey(396)
    x = jax.random.normal(rng, (4, 3, 2))
    cx = view_as_complex(x)
    assert jnp.all(x == view_as_real(cx))
    pt_x = to_pytorch(x)
    pt_cx = torch.view_as_complex(pt_x)
    assert jnp.all(cx == to_jax(pt_cx))
    assert jnp.all(view_as_real(cx) == to_jax(torch.view_as_real(pt_cx)))


def test_apply_rotary_embeddings():
    from gendo.llama.model_pt import apply_rotary_emb as pt_apply_rotary_embeddings
    rng = jax.random.PRNGKey(71)
    rng_q, rng_k = jax.random.split(rng, 2)
    batch_dim, seq_len, dim = 4, 3, 2
    xq = jax.random.normal(rng_q, (batch_dim, seq_len, dim))
    xk = jax.random.normal(rng_k, (batch_dim, seq_len, dim))
    freqs_cis = precompute_freqs_cis(dim, seq_len)
    out = apply_rotary_emb(xq, xk, freqs_cis)
    pt_out = pt_apply_rotary_embeddings(to_pytorch(xq), to_pytorch(xk), to_pytorch(freqs_cis))
    assert jnp.allclose(out[0], to_jax(pt_out[0]))