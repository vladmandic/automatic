from typing import Optional, Union, Tuple
import torch
import diffusers
import diffusers.utils.torch_utils


# copied from diffusers.PNDMScheduler._get_prev_sample
def PNDMScheduler__get_prev_sample(self, sample: torch.FloatTensor, timestep, prev_timestep, model_output):
    torch.dml.synchronize_tensor(sample) # DML synchronize
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    if self.config.prediction_type == "v_prediction":
        model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    elif self.config.prediction_type != "epsilon":
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon` or `v_prediction`"
        )

    sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

    model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
        alpha_prod_t * beta_prod_t * alpha_prod_t_prev
    ) ** (0.5)

    # full formula (9)
    prev_sample = (
        sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
    )

    return prev_sample


diffusers.PNDMScheduler._get_prev_sample = PNDMScheduler__get_prev_sample # pylint: disable=protected-access


# copied from diffusers.UniPCMultistepScheduler.multistep_uni_p_bh_update
def UniPCMultistepScheduler_multistep_uni_p_bh_update(
    self: diffusers.UniPCMultistepScheduler,
    model_output: torch.FloatTensor,
    *args,
    sample: torch.FloatTensor = None,
    order: int = None,
    **_,
) -> torch.FloatTensor:
    if sample is None:
        if len(args) > 1:
            sample = args[1]
        else:
            raise ValueError(" missing `sample` as a required keyward argument")
    if order is None:
        if len(args) > 2:
            order = args[2]
        else:
            raise ValueError(" missing `order` as a required keyward argument")
    model_output_list = self.model_outputs

    s0 = self.timestep_list[-1]
    m0 = model_output_list[-1]
    x = sample

    if self.solver_p:
        x_t = self.solver_p.step(model_output, s0, x).prev_sample
        return x_t

    torch.dml.synchronize_tensor(sample) # DML synchronize
    sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

    h = lambda_t - lambda_s0
    device = sample.device

    rks = []
    D1s = []
    for i in range(1, order):
        si = self.step_index - i
        mi = model_output_list[-(i + 1)]
        alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
        lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
        rk = (lambda_si - lambda_s0) / h
        rks.append(rk)
        D1s.append((mi - m0) / rk)

    rks.append(1.0)
    rks = torch.tensor(rks, device=device)

    R = []
    b = []

    hh = -h if self.predict_x0 else h
    h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
    h_phi_k = h_phi_1 / hh - 1

    factorial_i = 1

    if self.config.solver_type == "bh1":
        B_h = hh
    elif self.config.solver_type == "bh2":
        B_h = torch.expm1(hh)
    else:
        raise NotImplementedError

    for i in range(1, order + 1):
        R.append(torch.pow(rks, i - 1))
        b.append(h_phi_k * factorial_i / B_h)
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial_i

    R = torch.stack(R)
    b = torch.tensor(b, device=device)

    rhos_p = None
    if len(D1s) > 0:
        D1s = torch.stack(D1s, dim=1)  # (B, K)
        # for order 2, we use a simplified version
        if order == 2:
            rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
    else:
        D1s = None

    if self.predict_x0:
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res
    else:
        x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - sigma_t * B_h * pred_res

    x_t = x_t.to(x.dtype)
    return x_t


diffusers.UniPCMultistepScheduler.multistep_uni_p_bh_update = UniPCMultistepScheduler_multistep_uni_p_bh_update


# copied from diffusers.LCMScheduler.step
def LCMScheduler_step(
        self: diffusers.LCMScheduler,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[diffusers.schedulers.scheduling_lcm.LCMSchedulerOutput, Tuple]:
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    # 1. get previous step value
    prev_step_index = self.step_index + 1
    if prev_step_index < len(self.timesteps):
        prev_timestep = self.timesteps[prev_step_index]
    else:
        prev_timestep = timestep

    # 2. compute alphas, betas
    torch.dml.synchronize_tensor(sample) # DML synchronize
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    # 3. Get scalings for boundary conditions
    c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

    # 4. Compute the predicted original sample x_0 based on the model parameterization
    if self.config.prediction_type == "epsilon":  # noise-prediction
        predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif self.config.prediction_type == "sample":  # x-prediction
        predicted_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":  # v-prediction
        predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction` for `LCMScheduler`."
        )

    # 5. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        predicted_original_sample = self._threshold_sample(predicted_original_sample)
    elif self.config.clip_sample:
        predicted_original_sample = predicted_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 6. Denoise model output using boundary conditions
    denoised = c_out * predicted_original_sample + c_skip * sample

    # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
    # Noise is not used for one-step sampling.
    if len(self.timesteps) > 1:
        noise = diffusers.utils.torch_utils.randn_tensor(model_output.shape, generator=generator, device=model_output.device)
        prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
    else:
        prev_sample = denoised

    # upon completion increase step index by one
    self._step_index += 1

    if not return_dict:
        return (prev_sample, denoised)

    return diffusers.schedulers.scheduling_lcm.LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)


diffusers.LCMScheduler.step = LCMScheduler_step
