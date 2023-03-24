from transformers import FlaxCLIPModel
import flax.linen as nn
from flax.core.frozen_dict import freeze
from typing import Callable
import jax.numpy as jnp
import jax
from flax import traverse_util
import optax
from flax.training.train_state import TrainState



class Classifier(nn.Module):
    num_classes: int
    backbone: nn.Module

    @nn.compact
    def __call__(self, x):
        x = self.backbone(x).pooler_output
        x = nn.Dense(self.num_classes, name='head', kernel_init=nn.zeros)(x)
        return x


# Note: FlaxCLIPModel is not a Flax Module
def load_model():
    clip = FlaxCLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    module = clip.module # Extract the Flax Module
    variables = {'params': clip.params} # Extract the parameters
    return module, variables



def train_step(state: TrainState, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])




def main():
    clip, clip_variables = load_model()
    vision_model, vision_model_vars = clip.bind(clip_variables).vision_model.unbind()

    num_classes = 3
    model = Classifier(num_classes=num_classes, backbone=vision_model)

    x = jnp.empty((1, 224, 224, 3))
    variables = model.init(jax.random.PRNGKey(1), x)
    params = variables['params']

    params = params.unfreeze()
    params['backbone'] = vision_model_vars['params']
    params = freeze(params)

    partition_optimizers = {'trainable': optax.adam(5e-3), 'frozen': optax.set_to_zero()}
    param_partitions = freeze(
        traverse_util.path_aware_map(
            lambda path, v: 'frozen' if 'backbone' in path else 'trainable', params)
    )
    tx = optax.multi_transform(partition_optimizers, param_partitions)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


