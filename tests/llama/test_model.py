import jax
import jax.numpy as jnp
import torch
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from gendo.llama.tokenizer import Tokenizer
from gendo.llama.convert import pt2jax, jax2pt, fill_pytorch
from gendo.llama.model import (
    precompute_freqs_cis,
    view_as_complex,
    view_as_real,
    apply_rotary_emb,
    repeat_kv,
    ModelArgs,
    RMSNorm,
    Attention,
    FeedForward,
    TransformerBlock,
    Transformer,
)


# reduces memory per layer from 2.2Gb to 0.8Gb
MODEL_ARGS = ModelArgs(max_batch_size=1, max_seq_len=512)


def init_pseudo_distributed():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "gloo", init_method="file:///tmp/test", rank=0, world_size=1
        )
    if not model_parallel_is_initialized():
        initialize_model_parallel(1)


def compare_with_pytorch(module, pt_module, rng, *args):
    # jax
    variables = module.init(rng, *args)
    out = module.apply(variables, *args)
    # pytorch
    pt_args = map(jax2pt, args)
    pt_out = pt_module(*pt_args)
    assert jnp.all(out == pt2jax(pt_out))


def test_rmsnorm():
    from gendo.llama.model_pt import RMSNorm as PtRMSNorm

    batch_size, dim = (3, 4)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (batch_size, dim))
    compare_with_pytorch(RMSNorm(dim), PtRMSNorm(dim), rng, x)


def test_precompute_freqs_cis():
    from gendo.llama.model_pt import precompute_freqs_cis as pt_precompute_freqs_cis

    res = precompute_freqs_cis(32, 8)
    pt_res = pt_precompute_freqs_cis(32, 8)
    assert jnp.allclose(res, pt2jax(pt_res))


def test_complex_conversions():
    import torch

    rng = jax.random.PRNGKey(396)
    x = jax.random.normal(rng, (4, 3, 2))
    cx = view_as_complex(x)
    assert jnp.all(x == view_as_real(cx))
    pt_x = jax2pt(x)
    pt_cx = torch.view_as_complex(pt_x)
    assert jnp.all(cx == pt2jax(pt_cx))
    assert jnp.all(view_as_real(cx) == pt2jax(torch.view_as_real(pt_cx)))


def test_apply_rotary_embeddings():
    from gendo.llama.model_pt import apply_rotary_emb as pt_apply_rotary_embeddings

    rng = jax.random.PRNGKey(71)
    rng_q, rng_k = jax.random.split(rng, 2)
    batch_dim, seq_len, dim = 4, 3, 2
    xq = jax.random.normal(rng_q, (batch_dim, seq_len, dim))
    xk = jax.random.normal(rng_k, (batch_dim, seq_len, dim))
    freqs_cis = precompute_freqs_cis(dim, seq_len)
    out = apply_rotary_emb(xq, xk, freqs_cis)
    pt_out = pt_apply_rotary_embeddings(
        jax2pt(xq), jax2pt(xk), jax2pt(freqs_cis)
    )
    assert jnp.allclose(out[0], pt2jax(pt_out[0]))


def test_repeat_kv():
    from gendo.llama.model_pt import repeat_kv as pt_repeat_kw

    rng = jax.random.PRNGKey(134)
    x = jax.random.normal(rng, (5, 4, 3, 2))
    out = repeat_kv(x, 6)
    pt_x = jax2pt(x)
    pt_out = pt_repeat_kw(pt_x, 6)
    assert jnp.allclose(out, pt2jax(pt_out))


def test_attention():
    from gendo.llama.model_pt import Attention as PtAttention

    init_pseudo_distributed()

    args = MODEL_ARGS
    bsz, seqlen, dim = (1, 5, args.dim)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (bsz, seqlen, dim))

    freqs_cis = precompute_freqs_cis(128, seqlen)
    attn = Attention(args)
    variables = attn.init(rng, x, 0, freqs_cis, None)
    out, variable_updates = attn.apply(variables, x, 0, freqs_cis, None, mutable=["cache"])

    pt_attn = PtAttention(args)
    # fill_from_jax(pt_attn, variables["params"])
    params = variables["params"]
    pt_attn.wq.weight.data = jax2pt(params["wq"]["kernel"].T)
    pt_attn.wk.weight.data = jax2pt(params["wk"]["kernel"].T)
    pt_attn.wv.weight.data = jax2pt(params["wv"]["kernel"].T)
    pt_attn.wo.weight.data = jax2pt(params["wo"]["kernel"].T)
    pt_attn.cache_k = jax2pt(variables["cache"]["cache_k"])
    pt_attn.cache_v = jax2pt(variables["cache"]["cache_v"])

    pt_x = jax2pt(x)
    pt_freqs_cis = jax2pt(freqs_cis)
    pt_out = pt_attn(pt_x, 0, pt_freqs_cis, None)
    assert jnp.allclose(pt2jax(pt_out), out, atol=1e-2)
    assert jnp.allclose(pt2jax(pt_attn.cache_k), variable_updates["cache"]["cache_k"], atol=1e-2)
    assert jnp.allclose(pt2jax(pt_attn.cache_v), variables["cache"]["cache_v"], atol=1e-2)


def test_feedforward():
    from gendo.llama.model_pt import FeedForward as PtFeedForward

    init_pseudo_distributed()

    args = MODEL_ARGS
    bsz, seqlen, dim = (1, 5, args.dim)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (bsz, seqlen, dim))

    # freqs_cis = precompute_freqs_cis(128, seqlen)
    ff = FeedForward(args.dim, args.dim // 2, args.multiple_of, args.ffn_dim_multiplier)
    variables = ff.init(rng, x)
    out = ff.apply(variables, x)

    pt_ff = PtFeedForward(
        args.dim, args.dim // 2, args.multiple_of, args.ffn_dim_multiplier
    )
    params = variables["params"]
    pt_ff.w1.weight.data = jax2pt(params["w1"]["kernel"].T)
    pt_ff.w2.weight.data = jax2pt(params["w2"]["kernel"].T)
    pt_ff.w3.weight.data = jax2pt(params["w3"]["kernel"].T)

    pt_x = jax2pt(x)
    pt_out = pt_ff(pt_x)
    assert jnp.allclose(pt2jax(pt_out), out, atol=1e-2)


def test_transformerblock():
    from gendo.llama.model_pt import TransformerBlock as PtTransformerBlock

    init_pseudo_distributed()

    args = MODEL_ARGS
    bsz, seqlen, dim = (1, 5, args.dim)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (bsz, seqlen, dim))
    freqs_cis = precompute_freqs_cis(128, seqlen)
    block = TransformerBlock(0, args)
    variables = block.init(rng, x, 0, freqs_cis, None)
    out, variable_updates = block.apply(variables, x, 0, freqs_cis, None, mutable=["cache"])

    pt_block = PtTransformerBlock(0, args)
    fill_pytorch(pt_block, variables["params"])

    pt_x = jax2pt(x)
    pt_freqs_cis = jax2pt(freqs_cis)
    pt_out = pt_block(pt_x, 0, pt_freqs_cis, None)
    assert jnp.allclose(pt2jax(pt_out), out, atol=1e-2)
    assert jnp.allclose(
        pt2jax(pt_block.attention.cache_k), variable_updates["cache"]["attention"]["cache_k"], atol=1e-2
    )
    assert jnp.allclose(
        pt2jax(pt_block.attention.cache_v), variable_updates["cache"]["attention"]["cache_v"], atol=1e-2
    )


def test_transformer():
    from gendo.llama.tokenizer import Tokenizer
    from gendo.llama.model_pt import Transformer as PtTransformer

    init_pseudo_distributed()

    args = ModelArgs(max_batch_size=1, max_seq_len=512)
    tokenizer = Tokenizer(model_path="/data/llama/tokenizer.model")
    args.vocab_size = tokenizer.n_words
    tokens = tokenizer.encode("frankenstein walks into a bar", False, False)
    tokens = jnp.asarray(tokens).reshape(1, -1)
    rng = jax.random.PRNGKey(925)
    model = Transformer(args)
    variables = model.init(rng, tokens, 0)
    model = model.bind(variables)
    out, _variable_updates = model.apply(variables, tokens, 0, mutable=["cache"])

    pt_tokens = jax2pt(tokens)
    pt_model = PtTransformer(args)
    fill_pytorch(pt_model, variables["params"])
    pt_out = pt_model(pt_tokens, 0)
    assert jnp.allclose(pt2jax(pt_out), out, atol=1e-2)
