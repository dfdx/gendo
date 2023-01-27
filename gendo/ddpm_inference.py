from pathlib import Path
from datetime import datetime
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel


def main():
    base_dir = Path("./data/cppe-raw-results/")
    unet = UNet2DModel.from_pretrained(base_dir / "unet")
    scheduler = DDPMScheduler.from_config(base_dir / "scheduler")
    pipe = DDPMPipeline(unet, scheduler)
    pipe.to("cuda")
    out = pipe(batch_size=16)
    for img in out["images"]:
        img.save(f"data/results/{datetime.now()}.jpg")
