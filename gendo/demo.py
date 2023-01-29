import os
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxStableDiffusionPipeline

SHOW_DIR = "_data/output/show"
# MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_ID = "_data/output/gulnara"


def main():
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        MODEL_ID, revision="flax", dtype=jax.numpy.bfloat16
    )

    prompt = "a photo of a beautiful woman"

    prng_seed = jax.random.PRNGKey(0)
    num_inference_steps = 50

    num_samples = jax.device_count()
    prompt = num_samples * [prompt]
    prompt_ids = pipeline.prepare_inputs(prompt)

    # shard inputs and rng
    params = replicate(params)
    prng_seed = jax.random.split(prng_seed, jax.device_count())
    prompt_ids = shard(prompt_ids)

    os.makedirs(SHOW_DIR, exist_ok=True)
    seeds = jax.random.split(prng_seed, 8)
    for i in range(8):
        print(f"Generating {i}... ", end="")
        images = pipeline(prompt_ids, params, seeds[i], num_inference_steps, jit=True).images
        images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
        img = images[0]
        img_path = os.path.join(SHOW_DIR, f"{i}.jpg")
        print(f"Saving to {img_path}")
        img.save(img_path)