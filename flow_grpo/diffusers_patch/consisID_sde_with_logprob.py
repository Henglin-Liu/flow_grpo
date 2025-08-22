# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union, Tuple
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler

def index_for_timestep(timestep, schedule_timesteps):

    indices = (schedule_timesteps == timestep).nonzero()

    # The sigma index that is taken for the **very** first `step`
    # is always the second index (or the last index if there is only 1)
    # This way we can ensure we don't accidentally skip a sigma in
    # case we start in the middle of the denoising schedule (e.g. for image-to-image)
    pos = 1 if len(indices) > 1 else 0

    return indices[pos].item()

# def step(
#     self,
#     model_output: torch.Tensor,
#     old_pred_original_sample: torch.Tensor,
#     timestep: int,
#     timestep_back: int,
#     sample: torch.Tensor,
#     eta: float = 0.0,
#     use_clipped_model_output: bool = False,
#     generator=None,
#     variance_noise: Optional[torch.Tensor] = None,
#     return_dict: bool = False,
# ) -> Union[DDIMSchedulerOutput, Tuple]:
#     """
#     Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
#     process from the learned model outputs (most often the predicted noise).

#     Args:
#         model_output (`torch.Tensor`):
#             The direct output from learned diffusion model.
#         timestep (`float`):
#             The current discrete timestep in the diffusion chain.
#         sample (`torch.Tensor`):
#             A current instance of a sample created by the diffusion process.
#         eta (`float`):
#             The weight of noise for added noise in diffusion step.
#         use_clipped_model_output (`bool`, defaults to `False`):
#             If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
#             because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
#             clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
#             `use_clipped_model_output` has no effect.
#         generator (`torch.Generator`, *optional*):
#             A random number generator.
#         variance_noise (`torch.Tensor`):
#             Alternative to generating noise with `generator` by directly providing the noise for the variance
#             itself. Useful for methods such as [`CycleDiffusion`].
#         return_dict (`bool`, *optional*, defaults to `True`):
#             Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

#     Returns:
#         [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
#             If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
#             tuple is returned where the first element is the sample tensor.

#     """
#     if self.num_inference_steps is None:
#         raise ValueError(
#             "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
#         )

#     # See formulas (12) and (16) of DDIM paper https://huggingface.co/papers/2010.02502
#     # Ideally, read DDIM paper in-detail understanding

#     # Notation (<variable name> -> <name in paper>
#     # - pred_noise_t -> e_theta(x_t, t)
#     # - pred_original_sample -> f_theta(x_t, t) or x_0
#     # - std_dev_t -> sigma_t
#     # - eta -> η
#     # - pred_sample_direction -> "direction pointing to x_t"
#     # - pred_prev_sample -> "x_t-1"

#     # 1. get previous step value (=t-1)
#     prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

#     # 2. compute alphas, betas
#     alpha_prod_t = self.alphas_cumprod[timestep]
#     alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
#     alpha_prod_t_back = self.alphas_cumprod[timestep_back] if timestep_back is not None else None

#     beta_prod_t = 1 - alpha_prod_t

#     # 3. compute predicted original sample from predicted noise also called
#     # "predicted x_0" of formula (12) from https://huggingface.co/papers/2010.02502
#     # To make style tests pass, commented out `pred_epsilon` as it is an unused variable
#     if self.config.prediction_type == "epsilon":
#         pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
#         # pred_epsilon = model_output
#     elif self.config.prediction_type == "sample":
#         pred_original_sample = model_output
#         # pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
#     elif self.config.prediction_type == "v_prediction":
#         pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
#         # pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
#     else:
#         raise ValueError(
#             f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
#             " `v_prediction`"
#         )

#     h, r, lamb, lamb_next = self.get_variables(alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back)
#     mult = list(self.get_mult(h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back))
#     mult_noise = (1 - alpha_prod_t_prev) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5

#     noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
#     prev_sample = mult[0] * sample - mult[1] * pred_original_sample + mult_noise * noise

#     # add =============
#     dt = sigma_prev - sigma
#     prev_sample_mean = mult[0] * sample - mult[1] * pred_original_sample
#     log_prob = (
#         -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
#         - torch.log(std_dev_t * torch.sqrt(-1*dt))
#         - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
#     )

#     # mean along all but batch dimension
#     log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
#     # add =============

#     if old_pred_original_sample is None or prev_timestep < 0:
#         # Save a network evaluation if all noise levels are 0 or on the first step
#         return prev_sample, pred_original_sample
#     else:
#         denoised_d = mult[2] * pred_original_sample - mult[3] * old_pred_original_sample
#         noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
#         x_advanced = mult[0] * sample - mult[1] * denoised_d + mult_noise * noise

#         prev_sample = x_advanced

#     if not return_dict:
#         return (prev_sample, pred_original_sample)

#     return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
    
def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    step_index = [index_for_timestep(t,schedule_timesteps=self.timesteps) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    # TODO modify sde_step_with_logprob (sd3_sde_with_logprob.py), because CogVideoXDPMScheduler do not have sigmas variable
    sigmas = self.timesteps / max(self.timesteps)
    sigmas = torch.cat([sigmas, torch.tensor([0],device=sigmas.device)])
    sigma = sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = sigmas[1].item()
    dt = sigma_prev - sigma

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level
    
    # our sde
    prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
        - torch.log(std_dev_t * torch.sqrt(-1*dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(
        timestep.device
    )
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance

    
def ddim_step_with_logprob(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    noise_level: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> Union[DDIMSchedulerOutput, Tuple]:
    """
    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
            predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output` will have not effect.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
            can directly provide the noise for the variance itself. This is useful for methods such as
            CycleDiffusion. (https://arxiv.org/abs/2210.05559)
        return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

    Returns:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.

    """
    # assert isinstance(self, DDIMScheduler)
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = (
        timestep - self.config.num_train_timesteps // self.num_inference_steps
    )
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(
        sample.device
    )

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (
            beta_prod_t**0.5
        ) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    # if self.config.thresholding:
    #     pred_original_sample = self._threshold_sample(pred_original_sample)
    # elif self.config.clip_sample:
    #     pred_original_sample = pred_original_sample.clamp(
    #         -self.config.clip_sample_range, self.config.clip_sample_range
    #     )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = noise_level * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample_mean = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # log prob of prev_sample given prev_sample_mean and std_dev_t
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), log_prob , prev_sample_mean, std_dev_t

    
# def ddim_step_with_logprob(
#     self,
#     model_output: torch.FloatTensor,
#     timestep: int,
#     sample: torch.FloatTensor,
#     noise_level: float = 0.0,
#     use_clipped_model_output: bool = False,
#     generator=None,
#     prev_sample: Optional[torch.FloatTensor] = None,
#     old_pred_original_sample: Optional[torch.FloatTensor] = None,
#     timestep_back: Optional[int] = None,
# ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
#     """
#     Predict the sample at the previous timestep with log probability calculation, 
#     adapted for CogVideoXDPMScheduler's special features.
    
#     Args:
#         model_output: direct output from learned diffusion model
#         timestep: current discrete timestep in diffusion chain
#         sample: current instance of sample being created by diffusion process
#         eta: weight of noise for added noise in diffusion step
#         use_clipped_model_output: if True, compute "corrected" model_output
#         generator: random number generator
#         prev_sample: optional precomputed previous sample
#         old_pred_original_sample: previous prediction of x0 (for multi-step)
#         timestep_back: additional timestep reference for multi-step prediction
        
#     Returns:
#         tuple of (prev_sample, log_prob)
#     """
#     if self.num_inference_steps is None:
#         raise ValueError(
#             "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
#         )

#     # 1. get previous step value (=t-1)
#     prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
#     prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

#     # 2. compute alphas, betas with CogVideo's special handling
#     alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
#     alpha_prod_t_prev = torch.where(
#         prev_timestep.cpu() >= 0,
#         self.alphas_cumprod.gather(0, prev_timestep.cpu()),
#         self.final_alpha_cumprod,
#     )
#     alpha_prod_t_back = (
#         self.alphas_cumprod.gather(0, timestep_back.cpu()) 
#         if timestep_back is not None 
#         else None
#     )
    
#     # Broadcast to sample shape
#     alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
#     alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)
#     if alpha_prod_t_back is not None:
#         alpha_prod_t_back = _left_broadcast(alpha_prod_t_back, sample.shape).to(sample.device)

#     beta_prod_t = 1 - alpha_prod_t

#     # 3. compute predicted original sample
#     if self.config.prediction_type == "epsilon":
#         pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
#     elif self.config.prediction_type == "sample":
#         pred_original_sample = model_output
#     elif self.config.prediction_type == "v_prediction":
#         pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
#     else:
#         raise ValueError(
#             f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`"
#         )

#     # 4. Clip or threshold "predicted x_0"
#     # if self.config.thresholding:
#     #     pred_original_sample = self._threshold_sample(pred_original_sample)
#     # elif self.config.clip_sample:
#     #     pred_original_sample = pred_original_sample.clamp(
#     #         -self.config.clip_sample_range, self.config.clip_sample_range
#     #     )

#     # 5. Compute CogVideo-specific variables and multipliers
#     h, r, lamb, lamb_next = self.get_variables(alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back)
#     mult = list(self.get_mult(h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back))
#     mult_noise = (1 - alpha_prod_t_prev) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5

#     # 6. Generate noise if needed
#     if prev_sample is None:
#         noise = randn_tensor(
#             sample.shape, 
#             generator=generator, 
#             device=sample.device, 
#             dtype=sample.dtype
#         )
#         prev_sample_mean = mult[0] * sample - mult[1] * pred_original_sample
#         prev_sample = prev_sample_mean + mult_noise * noise
#     else:
#         prev_sample_mean = prev_sample  # When prev_sample is provided, we assume it's the mean

#     # 7. Advanced step with old prediction if available
#     if old_pred_original_sample is not None and prev_timestep >= 0:
#         denoised_d = mult[2] * pred_original_sample - mult[3] * old_pred_original_sample
#         noise = randn_tensor(
#             sample.shape, 
#             generator=generator, 
#             device=sample.device, 
#             dtype=sample.dtype
#         )
#         x_advanced_mean = mult[0] * sample - mult[1] * denoised_d
#         x_advanced = x_advanced_mean + mult_noise * noise
#         prev_sample = x_advanced
#         prev_sample_mean = x_advanced_mean

#     # 8. Compute log probability
#     std_dev = mult_noise
#     log_prob = (
#         -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev**2))
#         - torch.log(std_dev)
#         - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
#     )
#     # mean along all but batch dimension
#     log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

#     return prev_sample.type(sample.dtype), log_prob , prev_sample_mean, std_dev