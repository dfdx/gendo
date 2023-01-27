from typing import Union, List
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from transformers import CLIPTokenizer, FlaxCLIPTextModel
from diffusers import (
    FlaxStableDiffusionPipeline,
    FlaxAutoencoderKL,
    FlaxUNet2DConditionModel,
    FlaxDDPMScheduler,
)
from diffusers.models.vae_flax import FlaxEncoder
from dmtools.data import UnetTuningDataset, NumpyLoader, JaxDataset


def predict(pipe: FlaxStableDiffusionPipeline, params, prompt: Union[str, List[str]], rng=None):
    if isinstance(prompt, str):
        prompt = [prompt] * jax.device_count()
    if not rng:
        rng = jax.random.PRNGKey(0)
    if len(rng.shape) == 1:
        rng = jax.random.split(rng, jax.device_count())
    prompt_ids = pipe.prepare_inputs(prompt)
    prompt_ids = shard(prompt_ids)
    p_params = replicate(params)
    images = pipe(prompt_ids, p_params, rng, jit=True)[0]
    images = images.reshape((images.shape[1],) + images.shape[-3:])
    images = pipe.numpy_to_pil(images)
    return images


def predict_and_save(pipe: FlaxStableDiffusionPipeline, params, prompt, rng=None):
    from datetime import datetime
    images = predict(pipe, params, prompt, rng=rng)
    for img in images:
        img.save(f"output/show/{datetime.now()}.jpg")


def train():
    train_data_dir = "data/cppe-5"
    output_dir = "output/cppe-5-flax"
    object_name = "person in protective equipment"
    model_name = "CompVis/stable-diffusion-v1-4"
    batch_size=1
    seed = 74
    rng = jax.random.PRNGKey(seed)


    pipe, params = FlaxStableDiffusionPipeline.from_pretrained(
        model_name,
        revision="bf16",
        dtype=jnp.bfloat16,
    )
    text_encoder, vae, unet = pipe.text_encoder, pipe.vae, pipe.unet
    # we can't just take vae.encoder since it doesn't exist before setup() is called
    # (see the details at https://flax.readthedocs.io/en/latest/advanced_topics/module_lifecycle.html#setup-for-unbound-modules)
    # so we simply create the encoder the same way it's done in FlaxAutoencoderKL itself
    vae_encoder = FlaxEncoder(
        in_channels=vae.config.in_channels,
        out_channels=vae.config.latent_channels,
        down_block_types=vae.config.down_block_types,
        block_out_channels=vae.config.block_out_channels,
        layers_per_block=vae.config.layers_per_block,
        act_fn=vae.config.act_fn,
        norm_num_groups=vae.config.norm_num_groups,
        double_z=True,
        dtype=vae.dtype,
    )
    # vae_encoder.init(rng, img)

    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    dataset = JaxDataset(UnetTuningDataset(train_data_dir, tokenizer, object_name))
    loader = NumpyLoader(dataset, batch_size=batch_size, shuffle=True)

    batch = next(iter(loader))
    for epoch in range(1):
        for batch in loader:
            img = jnp.expand_dims(batch[0]["pixel_values"], 0)
            latents = vae_encoder.apply({"params": params["vae"]["encoder"]}, img).latent_dist.sample()


    # latents = (
    #                 vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
    #             )
    #             latents = latents * 0.18215

    #             # Sample noise that we'll add to the latents
    #             noise = torch.randn(latents.shape).to(latents.device)
    #             bsz = latents.shape[0]
    #             # Sample a random timestep for each image
    #             timesteps = torch.randint(
    #                 0,
    #                 noise_scheduler.config.num_train_timesteps,
    #                 (bsz,),
    #                 device=latents.device,
    #             ).long()

    #             # Add noise to the latents according to the noise magnitude at each timestep
    #             # (this is the forward diffusion process)
    #             noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    #             # Get the text embedding for conditioning
    #             encoder_hidden_states = text_encoder(batch["input_ids"])[0]

    #             # Predict the noise residual
    #             noise_pred = unet(
    #                 noisy_latents, timesteps, encoder_hidden_states
    #             ).sample

    #             loss = (
    #                 F.mse_loss(noise_pred, noise, reduction="none")
    #                 .mean([1, 2, 3])
    #                 .mean()
    #             )
    #             accelerator.backward(loss)

    #             # # Zero out the gradients for all token embeddings except the newly added
    #             # # embeddings for the concept, as we only want to optimize the concept embeddings
    #             # if accelerator.num_processes > 1:
    #             #     grads = text_encoder.module.get_input_embeddings().weight.grad
    #             # else:
    #             #     grads = text_encoder.get_input_embeddings().weight.grad
    #             # # Get the index for tokens that we want to zero the grads for
    #             # index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
    #             # grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()




def main():
    train_data_dir = "data/cppe-5"
    output_dir = "output/cppe-5-flax"
    # gradient_accumulation_steps = 1
    mixed_precision = "fp16"
    pretrained_model_name_or_path = BASE_MODEL
    use_auth_token = True
    learning_rate = 1e-4
    save_steps = 500
    repeats = 100
    seed = 1981
    resolution = 512
    train_batch_size = 1
    num_train_epochs = 100
    max_train_steps = 5000
    scale_lr = True
    lr_scheduler = "constant"
    lr_warmup_steps = 500
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08
    local_rank = -1

    logging_dir = os.path.join(output_dir, "logs")

    tokenizer = FlaxCLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_auth_token=use_auth_token,
    )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        use_auth_token=use_auth_token,
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", use_auth_token=use_auth_token
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", use_auth_token=use_auth_token
    )

    # Freeze vae and text_encoder
    freeze_params(vae.parameters())
    freeze_params(text_encoder.parameters())

    if scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),  # only optimize the unet
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    train_dataset = UnetTuningDataset(
        data_root=train_data_dir,
        tokenizer=tokenizer,
        size=resolution,
        repeats=repeats,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Move vae and unet to device
    vae.to(accelerator.device)
    unet.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # Keep vae and text_encoder in eval model as we don't train these
    vae.eval()
    text_encoder.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("stable_diffusion_tuning", config=vars(args))

    # Train!
    total_batch_size = (
        train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = (
                    vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                )
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )
                accelerator.backward(loss)

                # # Zero out the gradients for all token embeddings except the newly added
                # # embeddings for the concept, as we only want to optimize the concept embeddings
                # if accelerator.num_processes > 1:
                #     grads = text_encoder.module.get_input_embeddings().weight.grad
                # else:
                #     grads = text_encoder.get_input_embeddings().weight.grad
                # # Get the index for tokens that we want to zero the grads for
                # index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                # grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # if global_step % save_steps == 0:
                #     save_progress(text_encoder, placeholder_token_id, accelerator, args)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,
            scheduler=PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                skip_prk_steps=True,
            ),
            safety_checker=None,
            feature_extractor=CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            ),
        )
        pipeline.save_pretrained(output_dir)
        # Also save the newly trained embeddings
        # save_progress(text_encoder, placeholder_token_id, accelerator, args)

    accelerator.end_training()


if __name__ == "__main__" and "__file__" in globals():
    main()
