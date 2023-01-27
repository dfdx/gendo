# make sure you're logged in with `huggingface-cli login`
import os
from datetime import datetime
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


TOKEN = os.getenv("HUGGINGFACE_TOKEN")


def predict(pipe, prompt, n=4):
    with autocast("cuda"):
        images = pipe(prompt, num_images_per_prompt=n).images
        for img in images:
            img.save(f"output/show/{datetime.now()}.jpg")


pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    # "data/results/sd-ppe",
    use_auth_token=TOKEN,
    revision="fp16",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "small country house near the forest. hyper realistic. masterpiece"
predict(pipe, prompt, 4)

