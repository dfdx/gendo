import jax
import jax.numpy as jnp
import torch
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from gendo.llama.convert import to_jax, to_pytorch
from gendo.llama.model import (
    ModelArgs,
    RMSNorm,
    Attention,
    precompute_freqs_cis,
    view_as_complex,
    view_as_real,
    apply_rotary_emb,
    repeat_kv,
)


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
    pt_args = map(to_pytorch, args)
    pt_out = pt_module(*pt_args)
    assert jnp.all(out == to_jax(pt_out))


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
    assert jnp.allclose(res, to_jax(pt_res))


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
    pt_out = pt_apply_rotary_embeddings(
        to_pytorch(xq), to_pytorch(xk), to_pytorch(freqs_cis)
    )
    assert jnp.allclose(out[0], to_jax(pt_out[0]))


def test_repeat_kv():
    from gendo.llama.model_pt import repeat_kv as pt_repeat_kw

    rng = jax.random.PRNGKey(134)
    x = jax.random.normal(rng, (5, 4, 3, 2))
    out = repeat_kv(x, 6)
    pt_x = to_pytorch(x)
    pt_out = pt_repeat_kw(pt_x, 6)
    assert jnp.allclose(out, to_jax(pt_out))


def test_attention():
    from gendo.llama.model_pt import Attention as PtAttention

    init_pseudo_distributed()

    args = ModelArgs()
    bsz, seqlen, dim = (2, 5, args.dim)
    rng = jax.random.PRNGKey(925)
    x = jax.random.normal(rng, (bsz, seqlen, dim))
    cache = (
        jnp.zeros((16, 32, 32, 128)),
        jnp.zeros((16, 32, 32, 128)),
    )

    freqs_cis = precompute_freqs_cis(128, seqlen)
    attn = Attention(args)
    variables = attn.init(rng, cache, x, 0, freqs_cis, None)
    cache_out, out = attn.apply(variables, cache, x, 0, freqs_cis, None)

    pt_attn = PtAttention(args)
    # fill_from_jax(pt_attn, variables["params"])
    params = variables["params"]
    pt_attn.wq.weight.data = to_pytorch(params["wq"]["kernel"].T)
    pt_attn.wk.weight.data = to_pytorch(params["wk"]["kernel"].T)
    pt_attn.wv.weight.data = to_pytorch(params["wv"]["kernel"].T)
    pt_attn.wo.weight.data = to_pytorch(params["wo"]["kernel"].T)
    pt_attn.cache_k = to_pytorch(cache[0])
    pt_attn.cache_v = to_pytorch(cache[1])

    pt_x = to_pytorch(x)
    pt_freqs_cis = to_pytorch(freqs_cis)
    pt_out = pt_attn(pt_x, 0, pt_freqs_cis, None)
    assert jnp.allclose(to_jax(pt_out), out, atol=1e-2)
    assert jnp.allclose(to_jax(pt_attn.cache_k), cache_out[0], atol=1e-2)
    assert jnp.allclose(to_jax(pt_attn.cache_v), cache_out[1], atol=1e-2)
