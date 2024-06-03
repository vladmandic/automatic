from types import SimpleNamespace
import os
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from modules import shared, devices, processing, sd_models, errors, sd_hijack_hypertile, processing_vae, sd_models_compile, hidiffusion
from modules.processing_helpers import resize_hires, calculate_base_steps, calculate_hires_steps, calculate_refiner_steps, save_intermediate, update_sampler
from modules.processing_args import set_pipeline_args
from modules.onnx_impl import preprocess_pipeline as preprocess_onnx_pipeline, check_parameters_changed as olive_check_parameters_changed


debug = shared.log.trace if os.environ.get('SD_DIFFUSERS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: DIFFUSERS')


def process_diffusers(p: processing.StableDiffusionProcessing):
    debug(f'Process diffusers args: {vars(p)}')
    orig_pipeline = shared.sd_model
    results = []

    def is_txt2img():
        return sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.TEXT_2_IMAGE

    def is_refiner_enabled():
        return p.enable_hr and p.refiner_steps > 0 and p.refiner_start > 0 and p.refiner_start < 1 and shared.sd_refiner is not None

    def update_pipeline(sd_model, p: processing.StableDiffusionProcessing):
        """
        import diffusers
        if p.sag_scale > 0 and is_txt2img():
            update_sampler(shared.sd_model)
            supported = ['DDIMScheduler', 'PNDMScheduler', 'DDPMScheduler', 'DEISMultistepScheduler', 'UniPCMultistepScheduler', 'DPMSolverMultistepScheduler', 'DPMSolverSinlgestepScheduler']
            if hasattr(sd_model, 'sfast'):
                shared.log.warning(f'SAG incompatible compile mode: backend={shared.opts.cuda_compile_backend}')
            elif sd_model.scheduler.__class__.__name__ in supported:
                sd_model = sd_models.switch_pipe(diffusers.StableDiffusionSAGPipeline, sd_model)
                p.extra_generation_params["SAG scale"] = p.sag_scale
                p.task_args['sag_scale'] = p.sag_scale
            else:
                shared.log.warning(f'SAG incompatible scheduler: current={sd_model.scheduler.__class__.__name__} supported={supported}')
        """
        if sd_models.get_diffusers_task(sd_model) == sd_models.DiffusersTaskType.INPAINTING and getattr(p, 'image_mask', None) is None and p.task_args.get('image_mask', None) is None and getattr(p, 'mask', None) is None:
            shared.log.warning('Processing: mode=inpaint mask=None')
            sd_model = sd_models.set_diffuser_pipe(sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
        if shared.opts.cuda_compile_backend == "olive-ai":
            sd_model = olive_check_parameters_changed(p, is_refiner_enabled())
        if sd_model.__class__.__name__ == "OnnxRawPipeline":
            sd_model = preprocess_onnx_pipeline(p)
            nonlocal orig_pipeline
            orig_pipeline = sd_model # processed ONNX pipeline should not be replaced with original pipeline.
        if getattr(sd_model, "current_attn_name", None) != shared.opts.cross_attention_optimization:
            shared.log.info(f"Setting attention optimization: {shared.opts.cross_attention_optimization}")
            sd_models.set_diffusers_attention(sd_model)
        return sd_model

    # sanitize init_images
    if hasattr(p, 'init_images') and getattr(p, 'init_images', None) is None:
        del p.init_images
    if hasattr(p, 'init_images') and not isinstance(getattr(p, 'init_images', []), list):
        p.init_images = [p.init_images]
    if len(getattr(p, 'init_images', [])) > 0:
        while len(p.init_images) < len(p.prompts):
            p.init_images.append(p.init_images[-1])

    if shared.state.interrupted or shared.state.skipped:
        shared.sd_model = orig_pipeline
        return results

    # pipeline type is set earlier in processing, but check for sanity
    is_control = getattr(p, 'is_control', False) is True
    has_images = len(getattr(p, 'init_images' ,[])) > 0
    if sd_models.get_diffusers_task(shared.sd_model) != sd_models.DiffusersTaskType.TEXT_2_IMAGE and not has_images and not is_control:
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE) # reset pipeline
    if hasattr(shared.sd_model, 'unet') and hasattr(shared.sd_model.unet, 'config') and hasattr(shared.sd_model.unet.config, 'in_channels') and shared.sd_model.unet.config.in_channels == 9 and not is_control:
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.INPAINTING) # force pipeline
        if len(getattr(p, 'init_images', [])) == 0:
            p.init_images = [TF.to_pil_image(torch.rand((3, getattr(p, 'height', 512), getattr(p, 'width', 512))))]

    sd_models.move_model(shared.sd_model, devices.device)
    sd_models_compile.openvino_recompile_model(p, hires=False, refiner=False) # recompile if a parameter changes

    use_refiner_start = is_txt2img() and is_refiner_enabled() and not p.is_hr_pass and p.refiner_start > 0 and p.refiner_start < 1
    use_denoise_start = not is_txt2img() and p.refiner_start > 0 and p.refiner_start < 1

    shared.sd_model = update_pipeline(shared.sd_model, p)
    shared.log.info(f'Base: class={shared.sd_model.__class__.__name__}')
    update_sampler(p, shared.sd_model)
    base_args = set_pipeline_args(
        p=p,
        model=shared.sd_model,
        prompts=p.prompts,
        negative_prompts=p.negative_prompts,
        prompts_2=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else p.prompts,
        negative_prompts_2=[p.refiner_negative] if len(p.refiner_negative) > 0 else p.negative_prompts,
        num_inference_steps=calculate_base_steps(p, use_refiner_start=use_refiner_start, use_denoise_start=use_denoise_start),
        eta=shared.opts.scheduler_eta,
        guidance_scale=p.cfg_scale,
        guidance_rescale=p.diffusers_guidance_rescale,
        denoising_start=0 if use_refiner_start else p.refiner_start if use_denoise_start else None,
        denoising_end=p.refiner_start if use_refiner_start else 1 if use_denoise_start else None,
        output_type='latent' if hasattr(shared.sd_model, 'vae') else 'np',
        # output_type='pil',
        clip_skip=p.clip_skip,
        desc='Base',
    )
    shared.state.sampling_steps = base_args.get('prior_num_inference_steps', None) or base_args.get('num_inference_steps', None) or p.steps
    p.extra_generation_params['Pipeline'] = shared.sd_model.__class__.__name__
    if shared.opts.scheduler_eta is not None and shared.opts.scheduler_eta > 0 and shared.opts.scheduler_eta < 1:
        p.extra_generation_params["Sampler Eta"] = shared.opts.scheduler_eta
    output = None
    try:
        t0 = time.time()
        sd_models_compile.check_deepcache(enable=True)
        sd_models.move_model(shared.sd_model, devices.device)
        hidiffusion.apply_hidiffusion(p, shared.sd_model_type)
        # if 'image' in base_args:
        #    base_args['image'] = set_latents(p)
        if hasattr(shared.sd_model, 'tgate'):
            output = shared.sd_model.tgate(**base_args) # pylint: disable=not-callable
        else:
            output = shared.sd_model(**base_args)
        if isinstance(output, dict):
            output = SimpleNamespace(**output)
        hidiffusion.remove_hidiffusion(p)
        sd_models_compile.openvino_post_compile(op="base") # only executes on compiled vino models
        sd_models_compile.check_deepcache(enable=False)
        if shared.cmd_opts.profile:
            t1 = time.time()
            shared.log.debug(f'Profile: pipeline call: {t1-t0:.2f}')
        if not hasattr(output, 'images') and hasattr(output, 'frames'):
            if hasattr(output.frames[0], 'shape'):
                shared.log.debug(f'Generated: frames={output.frames[0].shape[1]}')
            else:
                shared.log.debug(f'Generated: frames={len(output.frames[0])}')
            output.images = output.frames[0]
        if isinstance(output.images, np.ndarray):
            output.images = torch.from_numpy(output.images)
    except AssertionError as e:
        shared.log.info(e)
    except ValueError as e:
        shared.state.interrupted = True
        shared.log.error(f'Processing: args={base_args} {e}')
        if shared.cmd_opts.debug:
            errors.display(e, 'Processing')
    except RuntimeError as e:
        shared.state.interrupted = True
        shared.log.error(f'Processing: args={base_args} {e}')
        errors.display(e, 'Processing')

    if hasattr(shared.sd_model, 'embedding_db') and len(shared.sd_model.embedding_db.embeddings_used) > 0: # register used embeddings
        p.extra_generation_params['Embeddings'] = ', '.join(shared.sd_model.embedding_db.embeddings_used)

    shared.state.nextjob()
    if shared.state.interrupted or shared.state.skipped:
        shared.sd_model = orig_pipeline
        return results

    # optional second pass
    if p.enable_hr:
        p.is_hr_pass = True
        p.init_hr(p.hr_scale, p.hr_upscaler, force=p.hr_force)
        prev_job = shared.state.job

        # hires runs on original pipeline
        if hasattr(shared.sd_model, 'restore_pipeline') and shared.sd_model.restore_pipeline is not None:
            shared.sd_model.restore_pipeline()

        # upscale
        if hasattr(p, 'height') and hasattr(p, 'width') and p.hr_upscaler is not None and p.hr_upscaler != 'None':
            shared.log.info(f'Upscale: upscaler="{p.hr_upscaler}" resize={p.hr_resize_x}x{p.hr_resize_y} upscale={p.hr_upscale_to_x}x{p.hr_upscale_to_y}')
            p.ops.append('upscale')
            if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_highres_fix and hasattr(shared.sd_model, 'vae'):
                save_intermediate(p, latents=output.images, suffix="-before-hires")
            shared.state.job = 'Upscale'
            output.images = resize_hires(p, latents=output.images)
            sd_hijack_hypertile.hypertile_set(p, hr=True)

        latent_upscale = shared.latent_upscale_modes.get(p.hr_upscaler, None)
        if (latent_upscale is not None or p.hr_force) and getattr(p, 'hr_denoising_strength', p.denoising_strength) > 0:
            p.ops.append('hires')
            sd_models_compile.openvino_recompile_model(p, hires=True, refiner=False)
            if shared.sd_model.__class__.__name__ == "OnnxRawPipeline":
                shared.sd_model = preprocess_onnx_pipeline(p)
            p.hr_force = True

        # hires
        if p.hr_force:
            shared.state.job_count = 2 * p.n_iter
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
            shared.log.info(f'HiRes: class={shared.sd_model.__class__.__name__} sampler="{p.hr_sampler_name}"')
            if p.is_control and hasattr(p, 'task_args') and p.task_args.get('image', None) is not None:
                if hasattr(shared.sd_model, "vae") and output.images is not None and len(output.images) > 0:
                    output.images = processing_vae.vae_decode(latents=output.images, model=shared.sd_model, full_quality=p.full_quality, output_type='pil') # controlnet cannnot deal with latent input
                    p.task_args['image'] = output.images # replace so hires uses new output
            sd_models.move_model(shared.sd_model, devices.device)
            orig_denoise = p.denoising_strength
            p.denoising_strength = getattr(p, 'hr_denoising_strength', p.denoising_strength)
            update_sampler(p, shared.sd_model, second_pass=True)
            hires_args = set_pipeline_args(
                p=p,
                model=shared.sd_model,
                prompts=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else p.prompts,
                negative_prompts=[p.refiner_negative] if len(p.refiner_negative) > 0 else p.negative_prompts,
                prompts_2=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else p.prompts,
                negative_prompts_2=[p.refiner_negative] if len(p.refiner_negative) > 0 else p.negative_prompts,
                num_inference_steps=calculate_hires_steps(p),
                eta=shared.opts.scheduler_eta,
                guidance_scale=p.image_cfg_scale if p.image_cfg_scale is not None else p.cfg_scale,
                guidance_rescale=p.diffusers_guidance_rescale,
                output_type='latent' if hasattr(shared.sd_model, 'vae') else 'np',
                clip_skip=p.clip_skip,
                image=output.images,
                strength=p.denoising_strength,
                desc='Hires',
            )
            shared.state.job = 'HiRes'
            shared.state.sampling_steps = hires_args.get('prior_num_inference_steps', None) or hires_args.get('num_inference_steps', None) or p.steps
            try:
                sd_models_compile.check_deepcache(enable=True)
                output = shared.sd_model(**hires_args) # pylint: disable=not-callable
                if isinstance(output, dict):
                    output = SimpleNamespace(**output)
                sd_models_compile.check_deepcache(enable=False)
                sd_models_compile.openvino_post_compile(op="base")
            except AssertionError as e:
                shared.log.info(e)
            p.denoising_strength = orig_denoise
        shared.state.job = prev_job
        shared.state.nextjob()
        p.is_hr_pass = False

    # optional refiner pass or decode
    if is_refiner_enabled():
        prev_job = shared.state.job
        shared.state.job = 'Refine'
        shared.state.job_count +=1
        if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_refiner and hasattr(shared.sd_model, 'vae'):
            save_intermediate(p, latents=output.images, suffix="-before-refiner")
        if shared.opts.diffusers_move_base:
            shared.log.debug('Moving to CPU: model=base')
            sd_models.move_model(shared.sd_model, devices.cpu)
        if shared.state.interrupted or shared.state.skipped:
            shared.sd_model = orig_pipeline
            return results
        if shared.opts.diffusers_move_refiner:
            sd_models.move_model(shared.sd_refiner, devices.device)
        p.ops.append('refine')
        p.is_refiner_pass = True
        sd_models_compile.openvino_recompile_model(p, hires=False, refiner=True)
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
        shared.sd_refiner = sd_models.set_diffuser_pipe(shared.sd_refiner, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
        for i in range(len(output.images)):
            image = output.images[i]
            noise_level = round(350 * p.denoising_strength)
            output_type='latent' if hasattr(shared.sd_refiner, 'vae') else 'np'
            if shared.sd_refiner.__class__.__name__ == 'StableDiffusionUpscalePipeline':
                image = processing_vae.vae_decode(latents=image, model=shared.sd_model, full_quality=p.full_quality, output_type='pil')
                p.extra_generation_params['Noise level'] = noise_level
                output_type = 'np'
            if hasattr(p, 'task_args') and p.task_args.get('image', None) is not None and output is not None: # replace input with output so it can be used by hires/refine
                p.task_args['image'] = image
            shared.log.info(f'Refiner: class={shared.sd_refiner.__class__.__name__}')
            update_sampler(p, shared.sd_refiner, second_pass=True)
            refiner_args = set_pipeline_args(
                p=p,
                model=shared.sd_refiner,
                prompts=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else p.prompts[i],
                negative_prompts=[p.refiner_negative] if len(p.refiner_negative) > 0 else p.negative_prompts[i],
                num_inference_steps=calculate_refiner_steps(p),
                eta=shared.opts.scheduler_eta,
                # strength=p.denoising_strength,
                noise_level=noise_level, # StableDiffusionUpscalePipeline only
                guidance_scale=p.image_cfg_scale if p.image_cfg_scale is not None else p.cfg_scale,
                guidance_rescale=p.diffusers_guidance_rescale,
                denoising_start=p.refiner_start if p.refiner_start > 0 and p.refiner_start < 1 else None,
                denoising_end=1 if p.refiner_start > 0 and p.refiner_start < 1 else None,
                image=image,
                output_type=output_type,
                clip_skip=p.clip_skip,
                desc='Refiner',
            )
            shared.state.sampling_steps = refiner_args.get('prior_num_inference_steps', None) or refiner_args.get('num_inference_steps', None) or p.steps
            try:
                if 'requires_aesthetics_score' in shared.sd_refiner.config: # sdxl-model needs false and sdxl-refiner needs true
                    shared.sd_refiner.register_to_config(requires_aesthetics_score = getattr(shared.sd_refiner, 'tokenizer', None) is None)
                refiner_output = shared.sd_refiner(**refiner_args) # pylint: disable=not-callable
                if isinstance(refiner_output, dict):
                    refiner_output = SimpleNamespace(**refiner_output)
                sd_models_compile.openvino_post_compile(op="refiner")
            except AssertionError as e:
                shared.log.info(e)

            if not shared.state.interrupted and not shared.state.skipped:
                refiner_images = processing_vae.vae_decode(latents=refiner_output.images, model=shared.sd_refiner, full_quality=True)
                for refiner_image in refiner_images:
                    results.append(refiner_image)

        if shared.opts.diffusers_move_refiner:
            shared.log.debug('Moving to CPU: model=refiner')
            sd_models.move_model(shared.sd_refiner, devices.cpu)
        shared.state.job = prev_job
        shared.state.nextjob()
        p.is_refiner_pass = False

    # final decode since there is no refiner
    if not is_refiner_enabled():
        if output is not None:
            if not hasattr(output, 'images') and hasattr(output, 'frames'):
                shared.log.debug(f'Generated: frames={len(output.frames[0])}')
                output.images = output.frames[0]
            if torch.is_tensor(output.images) and len(output.images) > 0 and any(s >= 512 for s in output.images.shape):
                results = output.images.cpu().numpy()
            elif hasattr(shared.sd_model, "vae") and output.images is not None and len(output.images) > 0:
                results = processing_vae.vae_decode(latents=output.images, model=shared.sd_model, full_quality=p.full_quality)
            elif hasattr(output, 'images'):
                results = output.images
            else:
                shared.log.warning('Processing returned no results')
                results = []
        else:
            shared.log.warning('Processing returned no results')
            results = []

    shared.sd_model = orig_pipeline
    return results
