import os
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxStableDiffusionPipeline

SHOW_DIR = "_data/output/show"
# MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_ID = "_data/output/gulnara"



def generate(pipe, params, prompt, prng_seed):
    num_inference_steps = 50

    num_samples = jax.device_count()
    prompt = num_samples * [prompt]
    prompt_ids = pipe.prepare_inputs(prompt)

    # shard inputs and rng
    params = replicate(params)
    prng_seed = jax.random.split(prng_seed, jax.device_count())
    # prng_seed = jax.random.split(prng_seed, num_samples)
    prompt_ids = shard(prompt_ids)

    os.makedirs(SHOW_DIR, exist_ok=True)
    for i in range(8):
        print(f"Generating {i}... ", end="")
        images = pipe(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
        images = pipe.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
        img = images[0]
        img_path = os.path.join(SHOW_DIR, f"{i}.jpg")
        print(f"Saving to {img_path}")
        img.save(img_path)
        # TODO: figure out how to pass multiple PRNG in one hop
        prng_seed = jax.random.split(prng_seed[0], 1)


def main():
    pipe, params = FlaxStableDiffusionPipeline.from_pretrained(
        MODEL_ID, revision="flax", dtype=jax.numpy.bfloat16
    )
    pipe.safety_checker = None

    prompt = "a photo of a beautiful woman with an unmbrella on a bridge. highly detailed, trending on artstation"

    prng_seed = jax.random.PRNGKey(1)
    generate(pipe, params, prompt, prng_seed)
