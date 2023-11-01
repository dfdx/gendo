from typing import Type
from multimethod import multimethod
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import torch
import torch.nn as tnn
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)


def to_pytorch(x):
    return torch.from_numpy(np.asarray(x).copy())


def to_jax(pt_x):
    return jnp.array(pt_x.detach().numpy())


@multimethod
def fill_pytorch(dst: ColumnParallelLinear | RowParallelLinear, params):
    dst.weight.data = to_pytorch(params["kernel"].T)
    assert "bias" not in params, "Bias is not supported yet"


@multimethod
def fill_pytorch(dst: tnn.Parameter, val):
    dst.data = to_pytorch(val)


@multimethod
def fill_pytorch(dst: tnn.Module, params):
    fields = params.keys()
    for field in fields:
        new_dst = getattr(dst, field)
        new_params = params[field]
        fill_pytorch(new_dst, new_params)
