# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
from typing import Any, Dict, List, Optional, Union
import torch

# from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.pipelines.consisid.pipeline_consisid import retrieve_timesteps,draw_kps
from .consisID_sde_with_logprob import sde_step_with_logprob,ddim_step_with_logprob
import inspect


@torch.no_grad()
def pipeline_with_logprob(
    self,
    image,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 6.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 226,
    skip_layer_guidance_scale: float = 2.8,
    noise_level: float = 0.7,
    num_frames: int = 49,
    eta: float = 0.0,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    id_vit_hidden: Optional[torch.Tensor] = None,
    id_cond: Optional[torch.Tensor] = None,
    rank: int=None,
):
    height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
    width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
    num_frames = num_frames or self.transformer.config.sample_frames    

    num_videos_per_prompt = 1
    
    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        image=image,
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        latents=latents,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    self._guidance_scale = guidance_scale
    self._current_timestep = None
    self._attention_kwargs = attention_kwargs
    self._interrupt = False


    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # 4. Prepare latent variables
    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas) # ([999, 932, 866, 799, 732, 666, 599, 532, 466, 399, 332, 266, 199, 132, 66], device='cuda:0')
    self._num_timesteps = len(timesteps)

    # 5. Prepare latents
    is_kps = getattr(self.transformer.config, "is_kps", False)
    kps_cond = kps_cond if is_kps else None
    if kps_cond is not None:
        kps_cond = draw_kps(image, kps_cond)
        kps_cond = self.video_processor.preprocess(kps_cond, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
    image = self.video_processor.preprocess(image, height=height, width=width).to(
        device, dtype=prompt_embeds.dtype
    )

    latent_channels = self.transformer.config.in_channels // 2

    latents, image_latents = self.prepare_latents(
        image, # torch.Size([1, 3, 480, 720])
        batch_size * num_videos_per_prompt, # 1
        latent_channels, # 16
        num_frames, # 49
        height, # 480
        width, # 720
        torch.float32, # prompt_embeds.dtype, # torch.float32, # torch.bfloat16
        device, # cuda:0
        generator,
        latents,
        kps_cond,
    ) # torch.Size([1, 13, 16, 60, 90]), torch.Size([1, 13, 16, 60, 90])

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if self.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # 5. Prepare timesteps
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []

    # 7. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        old_pred_original_sample = None
        for i, t in enumerate(timesteps): # tensor([999, 932, 866, 799, 732, 666, 599, 532, 466, 399, 332, 266, 199, 132, 66], device='cuda:0')
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            # if i == 10:
            #     print("====pipeline_with_prob====t:",t)
            #     print(f"[{rank}]latent_model_input:",latent_model_input[0][0][0])
            #     print(f"[{rank}]prompt_embeds:",prompt_embeds[0][0])
            #     print(f"[{rank}]timestep:",timestep)
            #     print(f"[{rank}]image_rotary_emb:",image_rotary_emb)
            #     print(f"[{rank}]id_vit_hidden:",id_vit_hidden[0][0][0])
            #     print(f"[{rank}]id_cond:",id_cond[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input, # torch.Size([4, 13, 32, 60, 90])
                encoder_hidden_states=prompt_embeds, # torch.Size([4, 226, 4096])
                timestep=timestep, # [999,999,999,999]
                image_rotary_emb=image_rotary_emb, # torch.Size([17550, 64]), torch.Size([17550, 64])
                attention_kwargs=attention_kwargs,
                return_dict=False,
                id_vit_hidden=id_vit_hidden, # list[5](torch.Size([2, 577, 1024]))
                id_cond=id_cond, # torch.Size([2, 1280])
            )[0]
            # if i == 10:
            #     print(f"[{rank}]noise_pred:",noise_pred[0][0])
            # noise_pred = noise_pred.float() # TODO


            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            # if i == 10:
            #     print(f"[{rank}][after guidance] noise_pred:",noise_pred[0][0])

            latents_dtype = latents.dtype

            # TODO !!!
            # latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            #     self.scheduler, 
            #     noise_pred.float(), 
            #     t.unsqueeze(0), 
            #     latents.float(), # torch.Size([1, 13, 16, 60, 90])
            #     noise_level=noise_level,
            # )
            # if i == 10:
            #     print(f"[{rank}][before ddim] noise_pred:",noise_pred[0][0])
            #     print(f"[{rank}][before ddim] latents:",latents[0][0][0])
            #     print(f"[{rank}][before ddim] noise_level:",noise_level)
            latents, log_prob,_,_ = ddim_step_with_logprob(
                self.scheduler, 
                noise_pred.float(), 
                t.unsqueeze(0), 
                latents.float(), # torch.Size([1, 13, 16, 60, 90])
                noise_level=noise_level,
            )
            # if i == 10:
            #     print(f"[{rank}][after ddim] noise_pred:",latents[0][0])
            #     print(f"[{rank}][after ddim] log_prob:",log_prob)
            # latents, old_pred_original_sample = self.scheduler.step(
            #     noise_pred,
            #     old_pred_original_sample,
            #     t,
            #     timesteps[i - 1] if i > 0 else None,
            #     latents,
            #     **extra_step_kwargs,
            #     return_dict=False,
            # )
            
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            all_latents.append(latents)
            all_log_probs.append(log_prob)
            
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    # replace =======
    # latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    # latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    # latents = latents.to(dtype=self.vae.dtype)
    # image = self.vae.decode(latents, return_dict=False)[0]
    # image = self.image_processor.postprocess(image, output_type=output_type)
    if not output_type == "latent": # GPU up!!
        video = self.decode_latents(latents)
        video = self.video_processor.postprocess_video(video=video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    return video, all_latents, all_log_probs
    
