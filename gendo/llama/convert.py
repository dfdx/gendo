# from typing import Type
# from multimethod import multimethod
import numpy as np
import jax.numpy as jnp
import torch


def to_pytorch(x):
    return torch.from_numpy(np.asarray(x).copy())


def to_jax(pt_x):
    return jnp.array(pt_x.detach().numpy())


def set_object_field(obj, path, val):
    x = obj
    for field in path[:-1]:
        x = getattr(x, field)
    setattr(x, path[-1], val)


# PT2JAX_NAMES = {
#     "weight": "kernel",
# }
# JAX2PT_NAMES = {
#     "kernel": "weight",
# }

# def fill_from_jax(pt_obj, params):
#     """
#     Given jax parameters `params`, fill PyTorch object `pt_obj`
#     """
#     for path, val in tree_util.tree_leaves_with_path(params):
#         path = [x.key for x in path] + ["data"]
#         path = [JAX2PT_NAMES.get(x, x) for x in path]
#         set_object_field(pt_obj, path, to_pytorch(val))


# from typing import Type
# from multimethod import multimethod
# import torch


# @multimethod
# def convert(in_t: Type[nn.Dense], out_t: Type[torch.nn.Linear], x: jax.Array):
#     print("jax to pytorch")


# @multimethod
# def convert(in_t: Type[torch.nn.Linear], out_t: Type[nn.Dense], x: torch.Tensor):
#     print("jax to pytorch")
