import argparse
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Optional, Any, Sequence
from functools import partial

import numpy as np
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset

import jax
import jax.numpy as jnp
import optax
import transformers
from flax.jax_utils import replicate, unreplicate
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

# just to avoid mistyping
TEXT_ENC = "text_encoder"
VAE = "vae"
UNET = "unet"
NOISE_SCHEDULER = "noise_scheduler"


# Cache compiled models across invocations of this script.
cc.initialize_cache(os.path.expanduser("~/.cache/jax/compilation_cache"))

logger = logging.getLogger(__name__)


# def get_params_to_save(params):
#     return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


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


def pipeline_and_params(models: dict, state: "TrainState", tokenizer):
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
        text_encoder=models[TEXT_ENC],
        vae=models[VAE],
        unet=models[UNET],
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
        safety_checker=safety_checker,
        feature_extractor=CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32"
        ),
    )
    def get_params_or_static(state: TrainState, name: str):
        if name in state.params:
            return state.params[name]
        else:
            return state.static[name]
    params = {
        "text_encoder": get_params_or_static(state, TEXT_ENC),
        "vae": get_params_or_static(state, VAE),
        "unet": get_params_or_static(state, UNET),
        "scheduler": noise_scheduler_state,
        "safety_checker": safety_checker.params,
    }
    return pipeline, params




class TrainState(train_state.TrainState):
    static: dict[str, Any]
    noise_scheduler_state: Any
    rng: Any


def create_train_state(
    components: dict,
    noise_scheduler_state,
    learning_rate: float = 5e-6,
    rng_seed: int = 0,
    trainables: Sequence[str] = ("unet",)
):
    # split components into trainable params and static arrays
    non_trainables = list(set(components.keys()) - set(trainables))
    params = {k: v for k, v in components.items() if k in trainables}
    static = {k: v for k, v in components.items() if k in non_trainables}

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
        static=static,
        tx=optimizer,
        noise_scheduler_state=noise_scheduler_state,
        rng=jax.random.PRNGKey(rng_seed)
    )


def get_params(name: str, static: dict, params: dict, trainables: Sequence[str]):
    if name in trainables:
        return params[name]
    else:
        return static[name]


def train_step(models: dict, state: TrainState, batch, trainables: Sequence[str] = ("unet",)):
    dropout_rng, sample_rng, new_rng = jax.random.split(state.rng, 3)
    text_encoder, vae, unet, noise_scheduler = (
        models["text_encoder"],
        models["vae"],
        models["unet"],
        models["noise_scheduler"],
    )

    def loss_fn(params):
        text_encoder_params = get_params("text_encoder", state.static, params, trainables)
        vae_params = get_params("vae", state.static, params, trainables)
        unet_params = get_params("unet", state.static, params, trainables)
        # Convert images to latent space
        vae_outputs = vae.apply(
            {"params": vae_params},
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

        encoder_hidden_states = text_encoder(
            batch["input_ids"],
            params=text_encoder_params,
            dropout_rng=dropout_rng,
            train=True,  # TODO: or not
        )[0]

        # Predict the noise residual
        model_pred = unet.apply(
            {"params": unet_params},
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
    loss, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, "batch")

    metrics = {"loss": loss}
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    # print(f"type(grads) == {type(grads)}")
    # print(f"grads[0].keys() == {grads[0].keys()}")
    # print(f"grads[1].keys() == {grads[1].keys()}")
    state = state.apply_gradients(grads=grads)
    state = state.replace(rng=new_rng)
    return state, metrics


def train(
    model_id: str,
    instance_data_dir: str,
    instance_prompt: str,
    trainables: Sequence[str] = ("unet",),
    revision: Optional[str] = "flax",
    seed: int = 1,
    batch_size: int = 1,
    learning_rate: float = 5e-6,
    num_train_epochs=50,
):
    set_seed(seed)

    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", revision=revision
    )
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

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    noise_scheduler_state = noise_scheduler.create_state()

    # Initialize our training
    # train_rngs = jax.random.split(rng, jax.local_device_count())

    state = create_train_state(
        {
            UNET: unet_params,
            TEXT_ENC: text_encoder.params,
            VAE: vae_params,
        },
        noise_scheduler_state,
        trainables=trainables,
        learning_rate=learning_rate,
    )
    models = {
        TEXT_ENC: text_encoder,
        VAE: vae,
        UNET: unet,
        NOISE_SCHEDULER: noise_scheduler,
    }

    p_state = jax_utils.replicate(state)
    p_train_step = jax.pmap(partial(train_step, models), "batch", static_broadcasted_argnums=2)

    # batch = next(iter(train_dataloader))
    # p_batch = shard(batch)
    # p_train_step(p_state, p_batch, trainables)

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
            p_state, metrics = p_train_step(p_state, batch, trainables)
            train_metrics.append(metrics)
            train_step_progress_bar.update(jax.local_device_count())
            # global_step += 1
        # train_metric = jax_utils.unreplicate(train_metric)
        train_step_progress_bar.close()
        epochs.write(
            f"Epoch... ({epoch + 1}/{num_train_epochs} | Loss: {train_metrics[-1]['loss']})"
        )
    # if jax.process_index() == 0:
    #     checkpoint(unet, unet_state.params, text_encoder, text_encoder_state.params, vae, vae_params, tokenizer, output_dir)
    return pipeline_and_params(models, unreplicate(p_state), tokenizer)


def generate(pipeline: FlaxStableDiffusionPipeline, params, prompts: list[str], prng_seed=None):
    pipeline.safety_checker = None
    if prng_seed is None:
        prng_seed = jax.random.PRNGKey(1)
    num_inference_steps = 50

    # num_samples = jax.device_count()
    # prompts = num_samples * [prompt]
    prompt_ids = pipeline.prepare_inputs(prompts)

    # shard inputs and rng
    p_params = replicate(params)
    prng_seed = jax.random.split(prng_seed, jax.device_count())
    # prng_seed = jax.random.split(prng_seed, num_samples)
    p_prompt_ids = shard(prompt_ids)

    SHOW_DIR = "_data/output/show"
    os.makedirs(SHOW_DIR, exist_ok=True)
    images = pipeline(
        p_prompt_ids, p_params, prng_seed, num_inference_steps, jit=True
    ).images
    images = pipeline.numpy_to_pil(
        np.asarray(images.reshape((len(prompts),) + images.shape[-3:]))
    )
    for i, img in enumerate(images):
        print(f"Generating {i}... ", end="")
        img_path = os.path.join(SHOW_DIR, f"{i}.jpg")
        print(f"Saving to {img_path}")
        img.save(img_path)

# def generate(pipeline, params, prompt, prng_seed=None):
#     pipeline.safety_checker = None
#     if prng_seed is None:
#         prng_seed = jax.random.PRNGKey(1)
#     num_inference_steps = 50

#     num_samples = jax.device_count()
#     prompt = num_samples * [prompt]
#     prompt_ids = pipeline.prepare_inputs(prompt)

#     # shard inputs and rng
#     params = replicate(params)
#     prng_seed = jax.random.split(prng_seed, jax.device_count())
#     # prng_seed = jax.random.split(prng_seed, num_samples)
#     prompt_ids = shard(prompt_ids)

#     SHOW_DIR = "_data/output/show"
#     os.makedirs(SHOW_DIR, exist_ok=True)
#     for i in range(8):
#         print(f"Generating {i}... ", end="")
#         images = pipeline(
#             prompt_ids, params, prng_seed, num_inference_steps, jit=True
#         ).images
#         images = pipeline.numpy_to_pil(
#             np.asarray(images.reshape((num_samples,) + images.shape[-3:]))
#         )
#         img = images[0]
#         img_path = os.path.join(SHOW_DIR, f"{i}.jpg")
#         print(f"Saving to {img_path}")
#         img.save(img_path)
#         # TODO: figure out how to pass multiple PRNG in one hop
#         prng_seed = jax.random.split(prng_seed[0], 1)


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
    trainables = ("unet",)
    revision = "flax"
    seed = 1
    batch_size = 1
    learning_rate: float = 5e-6
    num_train_epochs = 50
    model = FlaxStableDiffusion.fit(
        model_id,
        instance_data_dir,
        instance_prompt,
        trainables=trainables,
        revision=revision,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
    )
    model.save(output_dir)
    model = FlaxStableDiffusion.load(output_dir)
    model.predict("a photo of a beautiful woman")
