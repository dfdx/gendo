from typing import Callable
from tqdm import tqdm

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset
from jax.tree_util import tree_map, tree_leaves
from flax import traverse_util
from flax.core.frozen_dict import freeze
from flax.training.train_state import TrainState
from transformers import FlaxCLIPModel, CLIPImageProcessor

from jax.config import config

config.update("jax_debug_nans", True)


class Classifier(nn.Module):
    num_classes: int
    backbone: nn.Module

    @nn.compact
    def __call__(self, x):
        x = self.backbone(x).pooler_output
        x = nn.Dense(self.num_classes, name="head", kernel_init=nn.zeros)(x)
        return x


# Note: FlaxCLIPModel is not a Flax Module
def load_model():
    clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    module = clip.module  # Extract the Flax Module
    variables = {"params": clip.params}  # Extract the parameters
    return module, variables


@jax.jit
def train_step(state: TrainState, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main():
    batch_size = 16
    ds = load_dataset("food101", split="train[:5000]").with_format("jax")

    clip, clip_variables = load_model()
    vision_model, vision_model_vars = clip.bind(clip_variables).vision_model.unbind()

    num_classes = len(ds.unique("label"))
    model = Classifier(num_classes=num_classes, backbone=vision_model)

    x = jnp.empty((batch_size, 224, 224, 3))
    variables = model.init(jax.random.PRNGKey(1), x)
    params = variables["params"]

    params = params.unfreeze()
    params["backbone"] = vision_model_vars["params"]
    params = freeze(params)

    partition_optimizers = {
        # optax.clip_by_global_norm(1.0),
        "trainable": optax.chain(optax.adam(5e-4)),
        "frozen": optax.set_to_zero(),
    }
    param_partitions = freeze(
        traverse_util.path_aware_map(
            lambda path, v: "frozen" if "backbone" in path else "trainable", params
        )
    )
    tx = optax.multi_transform(partition_optimizers, param_partitions)

    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    proc = CLIPImageProcessor()
    unique_labels = ds.unique("label")
    label2idx = {l: i for i, l in enumerate(unique_labels)}
    for epoch in range(10):
        print(f"Epoch {epoch}", end="")
        losses = []
        pbar = tqdm(ds.iter(batch_size=batch_size), total=len(ds) // batch_size)
        for i, batch_ in enumerate(pbar):
            imgs, labels = batch_["image"], batch_["label"]
            batch = {
                "image": jnp.moveaxis(
                    jnp.stack(proc(imgs)["pixel_values"], axis=0), 1, -1
                ),
                "label": jnp.array([label2idx[lbl.item()] for lbl in labels]),
            }
            state, loss = train_step(state, batch)
            losses.append(loss.item())
            # if i % 500 // batch_size == 0:
            pbar.set_description(f"loss = {jnp.array(losses).mean()}")
