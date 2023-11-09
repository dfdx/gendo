from multimethod import multimethod
import re
import numpy as np
import jax
import jax.numpy as jnp
import torch
import torch.nn as tnn
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
)


###############################################################################
#                              dtype conversions                              #
###############################################################################

def pt2jax_dtype(pt_dtype: torch.dtype):
    if not isinstance(pt_dtype, torch.dtype):
        raise ValueError(f"The argument to to_jax_dtype() must be an instance of " +
                         f"torch.dtype, but instead {type(pt_dtype)} was received")
    # not using dicts because dtypes don't have stable hash
    if pt_dtype == torch.float32:
        return jnp.float32
    if pt_dtype == torch.float16:
        return jnp.float16
    if pt_dtype == torch.bfloat16:
        return jnp.bfloat16
    else:
        raise ValueError(f"Converting {pt_dtype} to a JAX type is not implemented")


def jax2pt_dtype(dtype: jnp.dtype):
    if not isinstance(dtype, jnp.dtype):
        raise ValueError(f"The argument to to_pytorch_dtype() must be an instance of " +
                         f"jnp.dtype, but instead {type(dtype)} was received")
    # not using dicts because dtypes don't have stable hash
    if dtype == jnp.float32:
        return torch.float32
    if dtype == jnp.float16:
        return torch.float16
    if dtype == jnp.bfloat16:
        return torch.bfloat16
    else:
        raise ValueError(f"Converting {dtype} to a PyTorch type is not implemented")


###############################################################################
#                              array conversions                              #
###############################################################################

def jax2pt(x: jax.Array):
    if x.dtype == jnp.bfloat16:
        # convert via fp32 because numpy doesn't support bf16
        x32 = x.astype(jnp.float32)
        return torch.from_numpy(np.asarray(x32).copy()).bfloat16()
    else:
        return torch.from_numpy(np.asarray(x).copy())


def pt2jax(pt_x: torch.Tensor):
    if pt_x.dtype == torch.bfloat16:
        # convert via fp32 because numpy doesn't support bf16
        pt_x32 = pt_x.to(torch.float32)
        return jnp.array(pt_x32.detach().numpy(), dtype=jnp.bfloat16)
    else:
        return jnp.array(pt_x.detach().numpy())


###############################################################################
#                filling PyTorch objects with JAX params                      #
###############################################################################

@multimethod
def fill_pytorch(dst: ColumnParallelLinear | RowParallelLinear, params):
    dst.weight.data = jax2pt(params["kernel"].T)
    assert "bias" not in params, "Bias is not supported yet"


@multimethod
def fill_pytorch(dst: ParallelEmbedding, params):
    dst.weight.data = jax2pt(params["embedding"])


@multimethod
def fill_pytorch(dst: tnn.Parameter, val):
    dst.data = jax2pt(val)


@multimethod
def fill_pytorch(dst: tnn.Module, params):
    fields = params.keys()
    for field in fields:
        if m := re.match(r"^([a-zA-Z0-9_]+)_([0-9]+)$", field):
            # match lists, e.g. layer_0
            pt_field, index_ = m.groups()
            index = int(index_)
            new_dst = getattr(dst, pt_field)[index]
        else:
            new_dst = getattr(dst, field)
        new_params = params[field]
        fill_pytorch(new_dst, new_params)
