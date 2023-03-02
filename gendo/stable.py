import argparse
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Optional, Any
from functools import partial

import numpy as np
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset

import jax
import jax.numpy as jnp
import optax
import transformers
from flax.jax_utils import replicate
from flax import linen as nn
from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.utils import check_min_version
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from jax.experimental.compilation_cache import compilation_cache as cc
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    CLIPFeatureExtractor,
    CLIPTokenizer,
    FlaxCLIPTextModel,
    set_seed,
)

from gendo.data import DreamBoothDataset, PromptDataset, collate_with_tokenizer


# Cache compiled models across invocations of this script.
cc.initialize_cache(os.path.expanduser("~/.cache/jax/compilation_cache"))

logger = logging.getLogger(__name__)


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def create_dataloader(
    instance_data_dir: str, instance_prompt: str, tokenizer, batch_size
):
    train_dataset = DreamBoothDataset(
        instance_data_root=instance_data_dir,
        instance_prompt=instance_prompt,
        tokenizer=tokenizer,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_with_tokenizer, tokenizer),
        drop_last=True,
    )
    return train_dataloader


class TrainState(train_state.TrainState):
    noise_scheduler_state: Any
    train_text_encoder: bool


def create_train_state(
    unet_params,
    text_encoder_params,
    vae_params,
    noise_scheduler_state,
    train_text_encoder: bool = False,
    learning_rate: float = 5e-6,
):
    params = {
        "text_encoder": text_encoder_params,
        "unet": unet_params,
        "vae": vae_params,
    }
    # Optimization
    constant_scheduler = optax.constant_schedule(learning_rate)
    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=0.9,
        b2=0.999,
        eps=1e-08,
        weight_decay=1e-2,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        adamw,
    )
    return TrainState.create(
        apply_fn=None,
        params=params,
        tx=optimizer,
        train_text_encoder=train_text_encoder,
        noise_scheduler_state=noise_scheduler_state,
    )


def train_step(models, state, batch, train_rng):
    dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)
    text_encoder, vae, unet, noise_scheduler = (
        models["text_encoder"],
        models["vae"],
        models["unet"],
        models["noise_scheduler"],
    )

    def loss_fn(params):
        # Convert images to latent space
        vae_outputs = vae.apply(
            {"params": params["vae"]},
            batch["pixel_values"],
            deterministic=True,
            method=vae.encode,
        )
        latents = vae_outputs.latent_dist.sample(sample_rng)
        # (NHWC) -> (NCHW)
        latents = jnp.transpose(latents, (0, 3, 1, 2))
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise_rng, timestep_rng = jax.random.split(sample_rng)
        noise = jax.random.normal(noise_rng, latents.shape)
        # Sample a random timestep for each image
        bsz = latents.shape[0]
        timesteps = jax.random.randint(
            timestep_rng,
            (bsz,),
            0,
            noise_scheduler.config.num_train_timesteps,
        )

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(
            state.noise_scheduler_state, latents, noise, timesteps
        )

        encoder_hidden_states = models["text_encoder"](
            batch["input_ids"],
            params=params["text_encoder"],
            dropout_rng=dropout_rng,
            train=True,
        )[0]
        # if state.train_text_encoder:
        #     encoder_hidden_states = text_encoder_state.apply_fn(
        #         batch["input_ids"],
        #         params=params["text_encoder"],
        #         dropout_rng=dropout_rng,
        #         train=True,
        #     )[0]
        # else:
        #     encoder_hidden_states = text_encoder(
        #         batch["input_ids"], params=text_encoder_state.params, train=False
        #     )[0]

        # Predict the noise residual
        model_pred = unet.apply(
            {"params": params["unet"]},
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            train=True,
        ).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(
                state.noise_scheduler_state, latents, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

        loss = (target - model_pred) ** 2
        loss = loss.mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)  # TODO: use only trainable parameters
    grad = jax.lax.pmean(grad, "batch")

    # new_unet_state = unet_state.apply_gradients(grads=grad["unet"])
    # if train_text_encoder:
    #     new_text_encoder_state = text_encoder_state.apply_gradients(
    #         grads=grad["text_encoder"]
    #     )
    # else:
    #     new_text_encoder_state = text_encoder_state

    metrics = {"loss": loss}
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    # return new_unet_state, new_text_encoder_state, metrics, new_train_rng
    return state


# class LatentDiffusionTrainer(nn.Module):

#     @nn.compact
#     def __call__(self, batch):


def pipeline_and_params(
    unet,
    unet_params,
    text_encoder,
    text_encoder_params,
    vae,
    vae_params,
    noise_scheduler,  # not used?
    noise_scheduler_state,  # not used?
    tokenizer,
):
    # Create the pipeline using the trained modules and save it.
    # surprise, surprise! pipeline._generate doesn't work with the DDPM scheduler,
    # so we have to use another one here
    noise_scheduler, noise_scheduler_state = FlaxPNDMScheduler.from_pretrained(
        # TODO: parametrize with actual model ID?
        "CompVis/stable-diffusion-v1-4",
        subfolder="scheduler",
    )
    safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker", from_pt=True
    )
    pipeline = FlaxStableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
        safety_checker=safety_checker,
        feature_extractor=CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32"
        ),
    )
    params = {
        "text_encoder": get_params_to_save(text_encoder_params),
        "vae": get_params_to_save(vae_params),
        "unet": get_params_to_save(unet_params),
        "scheduler": noise_scheduler_state,
        "safety_checker": safety_checker.params,
    }
    return pipeline, params


def train(
    model_id: str,
    instance_data_dir: str,
    instance_prompt: str,
    revision: Optional[str] = "flax",
    seed: int = 1,
    batch_size: int = 1,
    learning_rate: float = 5e-6,
    train_text_encoder: bool = False,
    num_train_epochs=50,
):
    set_seed(seed)

    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", revision=revision
    )
    rng = jax.random.PRNGKey(seed)
    train_dataloader = create_dataloader(
        instance_data_dir, instance_prompt, tokenizer, batch_size=batch_size
    )
    weight_dtype = jnp.float32  # TODO: use float16 and .to_fp16() on parameters

    # Load models
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", dtype=weight_dtype, revision=revision
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        model_id,
        dtype=weight_dtype,
        subfolder="vae",
        revision=revision,
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", dtype=weight_dtype, revision=revision
    )

    # Optimization
    # constant_scheduler = optax.constant_schedule(learning_rate)
    # adamw = optax.adamw(
    #     learning_rate=constant_scheduler,
    #     b1=0.9,
    #     b2=0.999,
    #     eps=1e-08,
    #     weight_decay=1e-2,
    # )
    # optimizer = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     adamw,
    # )
    # unet_state = train_state.TrainState.create(
    #     apply_fn=unet.__call__, params=unet_params, tx=optimizer
    # )
    # text_encoder_state = train_state.TrainState.create(
    #     apply_fn=text_encoder.__call__, params=text_encoder.params, tx=optimizer
    # )
    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    noise_scheduler_state = noise_scheduler.create_state()

    # Initialize our training
    train_rngs = jax.random.split(rng, jax.local_device_count())

    state = create_train_state(
        unet_params,
        text_encoder.params,
        vae_params,
        noise_scheduler_state,
        learning_rate=learning_rate,
    )
    models = {
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler,
    }



    batch = next(iter(train_dataloader))

    # Create parallel version of the train step
    # p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0, 1))
    p_train_step = jax.pmap(partial(train_step, models), "batch")


    # Replicate the train state on each device
    # unet_state = jax_utils.replicate(unet_state)
    # text_encoder_state = jax_utils.replicate(text_encoder_state)
    # vae_params = jax_utils.replicate(vae_params)

    p_state = jax_utils.replicate(state)
    p_batch = shard(batch)
    p_train_step(p_state, p_batch, train_rngs)


    # global_step = 0

    total_train_batch_size = batch_size * jax.local_device_count()
    epochs = tqdm(range(num_train_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_metrics = []
        steps_per_epoch = len(train_dataloader.dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(
            total=steps_per_epoch, desc="Training...", position=1, leave=False
        )
        # train
        for batch in train_dataloader:
            batch = shard(batch)
            unet_state, text_encoder_state, train_metric, train_rngs = p_train_step(
                unet_state, text_encoder_state, vae_params, batch, train_rngs
            )
            train_metrics.append(train_metric)
            train_step_progress_bar.update(jax.local_device_count())
            # global_step += 1
        train_metric = jax_utils.unreplicate(train_metric)
        train_step_progress_bar.close()
        epochs.write(
            f"Epoch... ({epoch + 1}/{num_train_epochs} | Loss: {train_metric['loss']})"
        )
    # if jax.process_index() == 0:
    #     checkpoint(unet, unet_state.params, text_encoder, text_encoder_state.params, vae, vae_params, tokenizer, output_dir)
    return pipeline_and_params(
        unet,
        unet_state.params,
        text_encoder,
        text_encoder_state.params,
        vae,
        vae_params,
        noise_scheduler,
        noise_scheduler_state,
        tokenizer,
    )


def generate(pipeline, params, prompt, prng_seed=None):
    pipeline.safety_checker = None
    if prng_seed is None:
        prng_seed = jax.random.PRNGKey(1)
    num_inference_steps = 50

    num_samples = jax.device_count()
    prompt = num_samples * [prompt]
    prompt_ids = pipeline.prepare_inputs(prompt)

    # shard inputs and rng
    params = replicate(params)
    prng_seed = jax.random.split(prng_seed, jax.device_count())
    # prng_seed = jax.random.split(prng_seed, num_samples)
    prompt_ids = shard(prompt_ids)

    SHOW_DIR = "_data/output/show"
    os.makedirs(SHOW_DIR, exist_ok=True)
    for i in range(8):
        print(f"Generating {i}... ", end="")
        images = pipeline(
            prompt_ids, params, prng_seed, num_inference_steps, jit=True
        ).images
        images = pipeline.numpy_to_pil(
            np.asarray(images.reshape((num_samples,) + images.shape[-3:]))
        )
        img = images[0]
        img_path = os.path.join(SHOW_DIR, f"{i}.jpg")
        print(f"Saving to {img_path}")
        img.save(img_path)
        # TODO: figure out how to pass multiple PRNG in one hop
        prng_seed = jax.random.split(prng_seed[0], 1)


class FlaxStableDiffusion:
    def __init__(self, pipeline, params):
        self.pipeline = pipeline
        self.params = params

    def predict(self, prompt, prng_seed=None):
        return generate(self.pipeline, self.params, prompt, prng_seed)

    @staticmethod
    def fit(*args, **kwargs):
        pipeline, params = train(*args, **kwargs)
        return FlaxStableDiffusion(pipeline, params)

    def save(self, path):
        self.pipeline.save_pretrained(path, params=self.params)

    @classmethod
    def load(cls, path):
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            path, revision="flax", dtype=jax.numpy.bfloat16
        )
        return FlaxStableDiffusion(pipeline, params)


def main():
    model_id = "runwayml/stable-diffusion-v1-5"
    instance_data_dir = "_data/input/gulnara"
    instance_prompt = "a photo of a beautiful woman"
    output_dir = "_data/models/gulnara"
    revision = "flax"
    seed = 1
    batch_size = 1
    learning_rate: float = 5e-6
    train_text_encoder = True
    num_train_epochs = 50
    model = FlaxStableDiffusion.fit(
        model_id,
        instance_data_dir,
        instance_prompt,
        revision=revision,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_text_encoder=train_text_encoder,
        num_train_epochs=num_train_epochs,
    )
    model.save(output_dir)
    model = FlaxStableDiffusion.load(output_dir)
    model.predict("a beautiful man with drums on a moon")
