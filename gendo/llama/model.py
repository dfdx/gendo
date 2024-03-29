import math
from dataclasses import dataclass
from typing import Optional, Tuple
from functools import partial

import jax
import jax.tree_util as tree_util
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
        return x * jax.lax.rsqrt(
            jnp.power(x, 2).mean(axis=-1, keepdims=True) + self.eps
        )

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


@partial(jax.jit, static_argnums=[0, 1])
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
    freqs = 1.0 / (
        theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype("float32") / dim)
    )
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


def repeat_kv(x: jax.Array, n_rep: int) -> jax.Array:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return jnp.tile(x[:, :, :, jnp.newaxis, :], (1, 1, 1, n_rep, 1)).reshape(
        bs, slen, n_kv_heads * n_rep, head_dim
    )


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        args (ModelArgs): Model configuration parameters.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Dense): Linear transformation for queries.
        wk (Dense): Linear transformation for keys.
        wv (Dense): Linear transformation for values.
        wo (Dense): Linear transformation for output.
        cache_k (jax.Array): Cached keys for attention.
        cache_v (jax.Array): Cached values for attention.
    """

    args: ModelArgs

    def setup(self):
        self.n_heads = self.args.n_heads
        self.n_kv_heads = (
            self.n_heads if self.args.n_kv_heads is None else self.args.n_kv_heads
        )

        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = self.args.dim // self.n_heads

        self.wq = nn.Dense(
            self.n_heads * self.head_dim,
            use_bias=False,
            # kernel_init=lambda x: x,
        )
        self.wk = nn.Dense(
            self.n_kv_heads * self.head_dim,
            use_bias=False,
            # kernel_init=lambda x: x,
        )
        self.wv = nn.Dense(
            self.n_kv_heads * self.head_dim,
            use_bias=False,
            # kernel_init=lambda x: x,
        )
        self.wo = nn.Dense(
            self.args.dim,
            use_bias=False,
        )
        self.cache_k = self.variable(
            "cache",
            "cache_k",
            jnp.zeros,
            (
                self.args.max_batch_size,
                self.args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            ),
            jnp.bfloat16,
        )
        self.cache_v = self.variable(
            "cache",
            "cache_v",
            jnp.zeros,
            (
                self.args.max_batch_size,
                self.args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            ),
            jnp.bfloat16,
        )

    def __call__(
        self,
        x: jax.Array,
        start_pos: int,
        freqs_cis: jax.Array,
        mask: Optional[jax.Array],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (jax.Array): Input array.
            start_pos (int): Starting position for caching.
            freqs_cis (jax.Array): Precomputed frequency array.
            mask (jax.Array, optional): Attention mask array.

        Returns:
            jax.Array: Output array after attention.
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis[start_pos : start_pos + x.shape[-2]])

        self.cache_k.value = self.cache_k.value.astype(xq.dtype)
        self.cache_v.value = self.cache_v.value.astype(xq.dtype)

        self.cache_k.value = self.cache_k.value.at[:bsz, start_pos : start_pos + seqlen].set(xk)
        self.cache_v.value = self.cache_v.value.at[:bsz, start_pos : start_pos + seqlen].set(xv)

        keys = self.cache_k.value[:bsz, : start_pos + seqlen]
        values = self.cache_v.value[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = jnp.moveaxis(xq, 1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = jnp.moveaxis(keys, 1, 2)
        values = jnp.moveaxis(values, 1, 2)

        scores = jnp.matmul(xq, jnp.moveaxis(keys, 2, 3)) / math.sqrt(self.head_dim)
        # scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        # scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = nn.softmax(scores.astype("float32"), axis=-1).astype(xq.dtype)
        output = jnp.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = jnp.moveaxis(output, 1, 2).ravel().reshape(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    multiple_of: int
    ffn_dim_multiplier: Optional[float]

    def setup(self):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (Linear): Linear transformation for the first layer.
            w2 (Linear): Linear transformation for the second layer.
            w3 (Linear): Linear transformation for the third layer.

        """
        hidden_dim = self.hidden_dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if self.ffn_dim_multiplier is not None:
            hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
        hidden_dim = self.multiple_of * (
            (hidden_dim + self.multiple_of - 1) // self.multiple_of
        )

        self.w1 = nn.Dense(
            hidden_dim,
            use_bias=False,  # gather_output=False, init_method=lambda x: x
        )
        self.w2 = nn.Dense(
            self.dim,
            use_bias=False,  # input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = nn.Dense(
            hidden_dim,
            use_bias=False,  # gather_output=False, init_method=lambda x: x
        )

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    layer_id: int
    args: ModelArgs

    def setup(self):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        args = self.args
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(
        self,
        x: jax.Array,
        start_pos: int,
        freqs_cis: jax.Array,
        mask: Optional[jax.Array],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (jax.Array): Input array.
            start_pos (int): Starting position for attention caching.
            freqs_cis (jax.Array): Precomputed cosine and sine frequencies.
            mask (jax.Array, optional): Masking tensor for attention. Defaults to None.

        Returns:
            jax.Array: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    args: ModelArgs

    def setup(self):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (nn.Embed): Token embeddings.
            layers (list): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (nn.Dense): Linear layer for final output.
            freqs_cis (jax.Array): Precomputed cosine and sine frequencies.

        """
        self.vocab_size = self.args.vocab_size
        self.n_layers = self.args.n_layers

        self.tok_embeddings = nn.Embed(self.args.vocab_size, self.args.dim)

        self.layers = [
            TransformerBlock(layer_id, self.args)
            for layer_id in range(self.args.n_layers)
        ]

        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.output = nn.Dense(self.args.vocab_size, use_bias=False)

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
        )

    def __call__(self, tokens: jax.Array, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (jax.Array): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            jax.Array: Output logits after applying the Transformer model.
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        # freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = jnp.full((1, 1, seqlen, seqlen), float("-inf"))
            mask = jnp.triu(mask, k=start_pos + 1).astype(h.dtype)
        for layer in self.layers:
            # note: unlike PyTorch implementation, we pass the full freqs_cis array
            # and take subarray later to avoid JIT errors due to dynamic shape
            h = layer(h, start_pos, self.freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).astype("float32")
        return output