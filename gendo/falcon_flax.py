from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import math
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange
from gendo.configuration_RW import RWConfig


def not_jax(fn):
    def inner(*args, **kwargs):
        raise Exception("This function has not been translated to JAX yet")
    return inner


def to_pytorch(x: jax.Array):
    return torch.tensor(np.asarray(x))


###############################################################################


def causal_mask(shape):
    mask = jnp.ones(shape, dtype=jnp.bool_)
    mask = jnp.tril(mask, 0)
    mask = jnp.where(~mask, -jnp.inf, mask)
    return mask


def causal_attention(q: jax.Array, k: jax.Array, v: jax.Array):
    """
    JAX version of torch.functional.scaled_dot_product_attention() with
    causal mask and zero dropout rate.
    """
    # q, k, v should have shape [batch..., seq_ken, head_dim]
    mask = causal_mask((q.shape[-2], k.shape[-2]))
    weight = nn.softmax((q @ jnp.moveaxis(k, -1, -2) / math.sqrt(q.shape[-1])) + mask, axis=-1)
    return weight @ v


# rotary pos emb helpers
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


class RotaryEmbedding(nn.Module):
    """
    Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is design to operate on queries and keys that are compatible with
    [batch_size, n_heads_per_partition, seq_len, head_dim] (e.g. MinGPTAttention format).
    """
    head_dim: int
    base: int = 10000

    def setup(self):
        self.inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))

    def cos_sin(
        self,
        seq_len: int,
    ) -> jax.Array:

        t = jnp.arange(seq_len)
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        cos_ = jnp.cos(emb)[None, :, :]
        sin_ = jnp.sin(emb)[None, :, :]

        return cos_, sin_

    def __call__(self, q, k):
        batch, seq_len, head_dim = q.shape
        cos, sin = self.cos_sin(seq_len)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


@not_jax
def _make_causal_mask(
    input_ids_shape, past_key_values_length: int
    # input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
):
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


@not_jax
def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def build_alibi_tensor(attention_mask: jax.Array, num_heads: int) -> jax.Array:
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = jnp.array(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))))
    powers = jnp.arange(1, 1 + closest_power_of_2, dtype=jnp.int32)
    slopes = jnp.power(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = jnp.array(2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))))
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = jnp.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=jnp.int32)
        slopes = jnp.concatenate([slopes, jnp.power(extra_base, extra_powers)], axis=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(axis=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor  # TODO: convert slopes to bfloat16?
    return alibi.reshape(batch_size * num_heads, 1, seq_length)


# def dropout_add(x: jax.Array, residual: jax.Array, prob: float, training: bool) -> jax.Array:
#     # out = F.dropout(x, p=prob, training=training)
#     out = nn.Dropout(rate=prob, deterministic=not training)(x)
#     out = residual + out
#     return out


class Attention(nn.Module):
    config: RWConfig

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = self.config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.maybe_rotary = RotaryEmbedding(self.config.head_dim) if self.config.rotary else lambda q, k: (q, k)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor

        self.query_key_value = nn.Dense(
            # self.hidden_size,
            3 * self.hidden_size if not self.config.multi_query else (self.hidden_size + 2 * self.head_dim),
            use_bias=self.config.bias,
        )
        self.multi_query = self.config.multi_query
        self.dense = nn.Dense(self.hidden_size, use_bias=self.config.bias)
        self.attention_dropout = nn.Dropout(self.config.attention_dropout)
        self.num_kv = self.config.n_head if not self.multi_query else 1

    def _split_heads(self, fused_qkv: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Split the last dimension into (num_heads, head_dim)

        Args:
            fused_qkv (`jax.Array`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if not self.multi_query:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.reshape(batch_size, seq_length, self.num_heads, 3, self.head_dim)
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.reshape(batch_size, seq_length, self.num_heads + 2, self.head_dim)
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

    def _merge_heads(self, x: jax.Array) -> jax.Array:
        """
        Merge heads together over the last dimenstion

        Args:
            x: (`jax.Array`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.reshape(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        # x = x.permute(0, 2, 1, 3)
        x = jnp.moveaxis(x, [0, 1, 2, 3], [0, 2, 1, 3])

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def __call__(
        self,
        hidden_states: jax.Array,
        alibi: jax.Array,
        attention_mask: jax.Array,
        layer_past: Optional[Tuple[jax.Array, jax.Array]] = None,
        head_mask: Optional[jax.Array] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        # query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        query_layer = jnp.moveaxis(query_layer, 1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = jnp.moveaxis(key_layer, 1, 2).reshape(
            batch_size * self.num_kv,
            q_length,
            self.head_dim,
        )
        # value_layer = value_layer.transpose(1, 2).reshape(batch_sCustom models groomingize * self.num_kv, q_length, self.head_dim)
        value_layer = jnp.moveaxis(value_layer, 1, 2).reshape(batch_size * self.num_kv, q_length, self.head_dim)

        query_layer, key_layer = self.maybe_rotary(query_layer, key_layer)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = jnp.concatenate((past_key, key_layer), axis=1)
            value_layer = jnp.concatenate((past_value, value_layer), axis=1)

        _, kv_length, _ = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        if alibi is None:
            query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
            key_layer_ = key_layer.reshape(batch_size, self.num_kv, -1, self.head_dim)
            value_layer_ = value_layer.reshape(batch_size, self.num_kv, -1, self.head_dim)

            attn_output = causal_attention(query_layer_, key_layer_, value_layer_)

            x = attn_output.reshape(batch_size, self.num_heads, q_length, self.head_dim)
            x = jnp.moveaxis(x, [0, 1, 2, 3], [0, 2, 1, 3])
            attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)

            output_tensor = self.dense(attn_output)

            outputs = (output_tensor, present)
            assert not output_attentions  # not supported.
            return outputs
        else:
            attention_mask_float = jnp.where(attention_mask, -1e9, attention_mask * 1.0).astype(jnp.bfloat16)
            matmul_result = query_layer @ jnp.moveaxis(key_layer, -1, -2)

            # change view to [batch_size, num_heads, q_length, kv_length]
            attention_scores = matmul_result.reshape(batch_size, self.num_heads, q_length, kv_length)

            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            input_dtype = attention_scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype == jnp.float16 or input_dtype == jnp.bfloat16:
                attention_scores = attention_scores.astype(jnp.float32)
            # attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
            attention_probs = nn.softmax(
                (attention_scores + alibi) * self.inv_norm_factor + attention_mask_float,
                axis=-1,
            )
            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attention_dropout(attention_probs, deterministic=True)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # change view [batch_size x num_heads, q_length, kv_length]
            attention_probs_reshaped = attention_probs.reshape(batch_size * self.num_heads, q_length, kv_length)

            # matmul: [batch_size * num_heads, q_length, head_dim]
            context_layer = attention_probs_reshaped @ value_layer

            # change view [batch_size, num_heads, q_length, head_dim]
            context_layer = self._merge_heads(context_layer)

            output_tensor = self.dense(context_layer)

            outputs = (output_tensor, present)
            if output_attentions:
                outputs += (attention_probs,)

            return outputs




class MLP(nn.Module):
    config: RWConfig

    def setup(self):
        super().__init__()
        hidden_size = self.config.hidden_size

        self.dense_h_to_4h = nn.Dense(4 * hidden_size, use_bias=self.config.bias)
        # self.act = nn.GELU()
        self.dense_4h_to_h = nn.Dense(hidden_size, use_bias=self.config.bias)
        self.hidden_dropout = self.config.hidden_dropout

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.gelu(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


class DecoderLayer(nn.Module):
    config: RWConfig

    def setup(self):
        config = self.config

        self.input_layernorm = nn.LayerNorm(epsilon=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = Attention(config)

        if not config.parallel_attn:
            # unused if parallel attn
            self.post_attention_layernorm = nn.LayerNorm(epsilon=config.layer_norm_epsilon)

        self.mlp = MLP(config)
        # TODO: pass via arguments
        training = True
        self.attn_dropout = nn.Dropout(rate=self.config.attention_dropout, deterministic=not training)
        self.hid_dropout = nn.Dropout(rate=self.config.hidden_dropout, deterministic=not training)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def __call__(
        self,
        hidden_states: jax.Array,
        alibi: jax.Array,
        attention_mask: jax.Array,
        layer_past: Optional[Tuple[jax.Array, jax.Array]] = None,
        head_mask: Optional[jax.Array] = None,
        # use_cache: bool = False,
        output_attentions: bool = False,
    ):

        layernorm_output = self.input_layernorm(hidden_states)
        residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            # use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        if not self.config.parallel_attn:
            # residual = dropout_add(attention_output, residual, self.config.attention_dropout, training=training)
            attention_output_do = self.attn_dropout(attention_output)
            residual = attention_output_do + residual
            layernorm_output = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        if self.config.parallel_attn:
            mlp_output += attention_output

        # output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=training)
        mlp_output_do = self.hid_dropout(mlp_output)
        output = mlp_output_do + residual

        # if use_cache:
        #     outputs = (output,) + outputs
        # else:
        outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions



def main():
    rng = jax.random.PRNGKey(1932)
    batch_size, seq_length, head_dim = 2, 5, 8
    self = RotaryEmbedding(head_dim)
    q = jax.random.normal(rng, (batch_size, seq_length, head_dim))
    k = jax.random.normal(rng, (batch_size, seq_length, head_dim))
    v = jax.random.normal(rng, (batch_size, seq_length, head_dim))
    variables = self.init(rng, q, k)

    hidden_size = 64
    config = RWConfig()
    hidden_states =  jax.random.normal(rng, (batch_size, seq_length, hidden_size))
    alibi = None
    attention_mask = jax.random.bernoulli(rng, shape=(seq_length, seq_length))
    self = Attention(config)
    variables = self.init(rng, hidden_states, alibi, attention_mask)
    self = self.bind(variables)

    tq, tk, tv = map(to_pytorch, (q, k, v))

    self = DecoderLayer(config)
    variables = self.init(rng, hidden_states, alibi, attention_mask)
    self = self.bind(variables, hidden_states, alibi, attention_mask)
    layer_past = None
    head_mask = None
    output_attentions = False
    decoder_out = self.apply(variables, hidden_states, alibi, attention_mask)