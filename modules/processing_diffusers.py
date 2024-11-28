from types import SimpleNamespace
import os
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from modules import shared, devices, processing, sd_models, errors, sd_hijack_hypertile, processing_vae, sd_models_compile, hidiffusion, timer, modelstats
from modules.processing_helpers import resize_hires, calculate_base_steps, calculate_hires_steps, calculate_refiner_steps, save_intermediate, update_sampler, is_txt2img, is_refiner_enabled
from modules.processing_args import set_pipeline_args
from modules.onnx_impl import preprocess_pipeline as preprocess_onnx_pipeline, check_parameters_changed as olive_check_parameters_changed


debug = shared.log.trace if os.environ.get('SD_DIFFUSERS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: DIFFUSERS')
last_p = None
orig_pipeline = shared.sd_model


def restore_state(p: processing.StableDiffusionProcessing):
    if p.state in ['reprocess_refine', 'reprocess_detail']:
        # validate
        if last_p is None:
            shared.log.warning(f'Restore state: op={p.state} last state missing')
            return p
        if p.__class__ != last_p.__class__:
            shared.log.warning(f'Restore state: op={p.state} last state is different type')
            return p
        if shared.history.count == 0:
            shared.log.warning(f'Restore state: op={p.state} last latents missing')
            return p
        state = p.state

        # set ops
        if state == 'reprocess_refine':
            # use new upscale values
            hr_scale, hr_upscaler, hr_resize_mode, hr_resize_context, hr_resize_x, hr_resize_y, hr_upscale_to_x, hr_upscale_to_y = p.hr_scale, p.hr_upscaler, p.hr_resize_mode, p.hr_resize_context, p.hr_resize_x, p.hr_resize_y, p.hr_upscale_to_x, p.hr_upscale_to_y # txt2img
            height, width, scale_by, resize_mode, resize_name, resize_context = p.height, p.width, p.scale_by, p.resize_mode, p.resize_name, p.resize_context # img2img
            p = last_p
            p.skip = ['encode', 'base']
            p.state = state
            p.enable_hr = True
            p.hr_force = True
            p.hr_scale, p.hr_upscaler, p.hr_resize_mode, p.hr_resize_context, p.hr_resize_x, p.hr_resize_y, p.hr_upscale_to_x, p.hr_upscale_to_y = hr_scale, hr_upscaler, hr_resize_mode, hr_resize_context, hr_resize_x, hr_resize_y, hr_upscale_to_x, hr_upscale_to_y
            p.height, p.width, p.scale_by, p.resize_mode, p.resize_name, p.resize_context = height, width, scale_by, resize_mode, resize_name, resize_context
            p.init_images = None
        if state == 'reprocess_detail':
            p.skip = ['encode', 'base', 'hires']
            p.detailer = True
        shared.log.info(f'Restore state: op={p.state} skip={p.skip}')
    return p


def process_base(p: processing.StableDiffusionProcessing):
    use_refiner_start = is_txt2img() and is_refiner_enabled(p) and not p.is_hr_pass and p.refiner_start > 0 and p.refiner_start < 1
    use_denoise_start = not is_txt2img() and p.refiner_start > 0 and p.refiner_start < 1

    shared.sd_model = update_pipeline(shared.sd_model, p)
    shared.log.info(f'Base: class={shared.sd_model.__class__.__name__}')
    update_sampler(p, shared.sd_model)
    timer.process.record('prepare')
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
        output_type='latent',
        clip_skip=p.clip_skip,
        desc='Base',
    )
    shared.state.sampling_steps = base_args.get('prior_num_inference_steps', None) or p.steps or base_args.get('num_inference_steps', None)
    if shared.opts.scheduler_eta is not None and shared.opts.scheduler_eta > 0 and shared.opts.scheduler_eta < 1:
        p.extra_generation_params["Sampler Eta"] = shared.opts.scheduler_eta
    output = None
    try:
        t0 = time.time()
        sd_models_compile.check_deepcache(enable=True)
        sd_models.move_model(shared.sd_model, devices.device)
        if hasattr(shared.sd_model, 'unet'):
            sd_models.move_model(shared.sd_model.unet, devices.device)
        if hasattr(shared.sd_model, 'transformer'):
            sd_models.move_model(shared.sd_model.transformer, devices.device)
        hidiffusion.apply(p, shared.sd_model_type)
        # if 'image' in base_args:
        #    base_args['image'] = set_latents(p)
        timer.process.record('move')
        if hasattr(shared.sd_model, 'tgate') and getattr(p, 'gate_step', -1) > 0:
            base_args['gate_step'] = p.gate_step
            output = shared.sd_model.tgate(**base_args) # pylint: disable=not-callable
        else:
            output = shared.sd_model(**base_args)
        if isinstance(output, dict):
            output = SimpleNamespace(**output)
        if isinstance(output, list):
            output = SimpleNamespace(images=output)
        if hasattr(output, 'images'):
            shared.history.add(output.images, info=processing.create_infotext(p), ops=p.ops)
        timer.process.record('pipeline')
        hidiffusion.unapply()
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
        err_args = base_args.copy()
        for k, v in base_args.items():
            if isinstance(v, torch.Tensor):
                err_args[k] = f'{v.device}:{v.dtype}:{v.shape}'
        shared.log.error(f'Processing: args={err_args} {e}')
        if shared.cmd_opts.debug:
            errors.display(e, 'Processing')
    except RuntimeError as e:
        shared.state.interrupted = True
        err_args = base_args.copy()
        for k, v in base_args.items():
            if isinstance(v, torch.Tensor):
                err_args[k] = f'{v.device}:{v.dtype}:{v.shape}'
        shared.log.error(f'Processing: step=base args={err_args} {e}')
        errors.display(e, 'Processing')
        modelstats.analyze()

    if hasattr(shared.sd_model, 'embedding_db') and len(shared.sd_model.embedding_db.embeddings_used) > 0: # register used embeddings
        p.extra_generation_params['Embeddings'] = ', '.join(shared.sd_model.embedding_db.embeddings_used)

    shared.state.nextjob()
    return output


def process_hires(p: processing.StableDiffusionProcessing, output):
    # optional second pass
    if p.enable_hr:
        p.is_hr_pass = True
        if hasattr(p, 'init_hr'):
            p.init_hr(p.hr_scale, p.hr_upscaler, force=p.hr_force)
        else:
            if not p.is_hr_pass: # fake hires for img2img if not actual hr pass
                p.hr_scale = p.scale_by
                p.hr_upscaler = p.resize_name
                p.hr_resize_mode = p.resize_mode
                p.hr_resize_context = p.resize_context
            p.hr_upscale_to_x = p.width * p.hr_scale if p.hr_resize_x == 0 else p.hr_resize_x
            p.hr_upscale_to_y = p.height * p.hr_scale if p.hr_resize_y == 0 else p.hr_resize_y
        prev_job = shared.state.job

        # hires runs on original pipeline
        if hasattr(shared.sd_model, 'restore_pipeline') and shared.sd_model.restore_pipeline is not None:
            shared.sd_model.restore_pipeline()

        # upscale
        if hasattr(p, 'height') and hasattr(p, 'width') and p.hr_resize_mode > 0 and (p.hr_upscaler != 'None' or p.hr_resize_mode == 5):
            shared.log.info(f'Upscale: mode={p.hr_resize_mode} upscaler="{p.hr_upscaler}" context="{p.hr_resize_context}" resize={p.hr_resize_x}x{p.hr_resize_y} upscale={p.hr_upscale_to_x}x{p.hr_upscale_to_y}')
            p.ops.append('upscale')
            if shared.opts.samples_save and not p.do_not_save_samples and shared.opts.save_images_before_highres_fix and hasattr(shared.sd_model, 'vae'):
                save_intermediate(p, latents=output.images, suffix="-before-hires")
            shared.state.job = 'Upscale'
            output.images = resize_hires(p, latents=output.images)
            sd_hijack_hypertile.hypertile_set(p, hr=True)

        latent_upscale = shared.latent_upscale_modes.get(p.hr_upscaler, None)
        strength = p.hr_denoising_strength if p.hr_denoising_strength > 0 else p.denoising_strength
        if (latent_upscale is not None or p.hr_force) and strength > 0:
            p.ops.append('hires')
            sd_models_compile.openvino_recompile_model(p, hires=True, refiner=False)
            if shared.sd_model.__class__.__name__ == "OnnxRawPipeline":
                shared.sd_model = preprocess_onnx_pipeline(p)
            p.hr_force = True

        # hires
        if p.hr_force and strength == 0:
            shared.log.warning('HiRes skip: denoising=0')
            p.hr_force = False
        if p.hr_force:
            shared.state.job_count = 2 * p.n_iter
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
            shared.log.info(f'HiRes: class={shared.sd_model.__class__.__name__} sampler="{p.hr_sampler_name}"')
            if 'Upscale' in shared.sd_model.__class__.__name__ or 'Flux' in shared.sd_model.__class__.__name__:
                output.images = processing_vae.vae_decode(latents=output.images, model=shared.sd_model, full_quality=p.full_quality, output_type='pil', width=p.width, height=p.height)
            if p.is_control and hasattr(p, 'task_args') and p.task_args.get('image', None) is not None:
                if hasattr(shared.sd_model, "vae") and output.images is not None and len(output.images) > 0:
                    output.images = processing_vae.vae_decode(latents=output.images, model=shared.sd_model, full_quality=p.full_quality, output_type='pil', width=p.hr_upscale_to_x, height=p.hr_upscale_to_y) # controlnet cannnot deal with latent input
                    p.task_args['image'] = output.images # replace so hires uses new output
            sd_models.move_model(shared.sd_model, devices.device)
            if hasattr(shared.sd_model, 'unet'):
                sd_models.move_model(shared.sd_model.unet, devices.device)
            if hasattr(shared.sd_model, 'transformer'):
                sd_models.move_model(shared.sd_model.transformer, devices.device)
            update_sampler(p, shared.sd_model, second_pass=True)
            orig_denoise = p.denoising_strength
            p.denoising_strength = strength
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
                output_type='latent',
                clip_skip=p.clip_skip,
                image=output.images,
                strength=strength,
                desc='Hires',
            )
            shared.state.job = 'HiRes'
            shared.state.sampling_steps = hires_args.get('prior_num_inference_steps', None) or p.steps or hires_args.get('num_inference_steps', None)
            try:
                sd_models_compile.check_deepcache(enable=True)
                output = shared.sd_model(**hires_args) # pylint: disable=not-callable
                if isinstance(output, dict):
                    output = SimpleNamespace(**output)
                if hasattr(output, 'images'):
                    shared.history.add(output.images, info=processing.create_infotext(p), ops=p.ops)
                sd_models_compile.check_deepcache(enable=False)
                sd_models_compile.openvino_post_compile(op="base")
            except AssertionError as e:
                shared.log.info(e)
            except RuntimeError as e:
                shared.state.interrupted = True
                shared.log.error(f'Processing step=hires: args={hires_args} {e}')
                errors.display(e, 'Processing')
                modelstats.analyze()
            p.denoising_strength = orig_denoise
        shared.state.job = prev_job
        shared.state.nextjob()
        p.is_hr_pass = False
        timer.process.record('hires')
    return output


def process_refine(p: processing.StableDiffusionProcessing, output):
    # optional refiner pass or decode
    if is_refiner_enabled(p):
        prev_job = shared.state.job
        shared.state.job = 'Refine'
        shared.state.job_count +=1
        if shared.opts.samples_save and not p.do_not_save_samples and shared.opts.save_images_before_refiner and hasattr(shared.sd_model, 'vae'):
            save_intermediate(p, latents=output.images, suffix="-before-refiner")
        if shared.opts.diffusers_move_base:
            shared.log.debug('Moving to CPU: model=base')
            sd_models.move_model(shared.sd_model, devices.cpu)
        if shared.state.interrupted or shared.state.skipped:
            shared.sd_model = orig_pipeline
            return output
        if shared.opts.diffusers_offload_mode == "balanced":
            shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        if shared.opts.diffusers_move_refiner:
            sd_models.move_model(shared.sd_refiner, devices.device)
            if hasattr(shared.sd_refiner, 'unet'):
                sd_models.move_model(shared.sd_model.unet, devices.device)
            if hasattr(shared.sd_refiner, 'transformer'):
                sd_models.move_model(shared.sd_model.transformer, devices.device)
        p.ops.append('refine')
        p.is_refiner_pass = True
        sd_models_compile.openvino_recompile_model(p, hires=False, refiner=True)
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
        shared.sd_refiner = sd_models.set_diffuser_pipe(shared.sd_refiner, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
        for i in range(len(output.images)):
            image = output.images[i]
            noise_level = round(350 * p.denoising_strength)
            output_type='latent'
            if 'Upscale' in shared.sd_refiner.__class__.__name__ or 'Flux' in shared.sd_refiner.__class__.__name__:
                image = processing_vae.vae_decode(latents=image, model=shared.sd_model, full_quality=p.full_quality, output_type='pil', width=p.width, height=p.height)
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
            shared.state.sampling_steps = refiner_args.get('prior_num_inference_steps', None) or p.steps or refiner_args.get('num_inference_steps', None)
            try:
                if 'requires_aesthetics_score' in shared.sd_refiner.config: # sdxl-model needs false and sdxl-refiner needs true
                    shared.sd_refiner.register_to_config(requires_aesthetics_score = getattr(shared.sd_refiner, 'tokenizer', None) is None)
                output = shared.sd_refiner(**refiner_args) # pylint: disable=not-callable
                if isinstance(output, dict):
                    output = SimpleNamespace(**output)
                if hasattr(output, 'images'):
                    shared.history.add(output.images, info=processing.create_infotext(p), ops=p.ops)
                sd_models_compile.openvino_post_compile(op="refiner")
            except AssertionError as e:
                shared.log.info(e)
            except RuntimeError as e:
                shared.state.interrupted = True
                shared.log.error(f'Processing step=refine: args={refiner_args} {e}')
                errors.display(e, 'Processing')
                modelstats.analyze()

            """ # TODO decode using refiner
            if not shared.state.interrupted and not shared.state.skipped:
                refiner_images = processing_vae.vae_decode(latents=refiner_output.images, model=shared.sd_refiner, full_quality=True, width=max(p.width, p.hr_upscale_to_x), height=max(p.height, p.hr_upscale_to_y))
                for refiner_image in refiner_images:
                    results.append(refiner_image)
            """

        if shared.opts.diffusers_offload_mode == "balanced":
            shared.sd_refiner = sd_models.apply_balanced_offload(shared.sd_refiner)
        elif shared.opts.diffusers_move_refiner:
            shared.log.debug('Moving to CPU: model=refiner')
            sd_models.move_model(shared.sd_refiner, devices.cpu)
        shared.state.job = prev_job
        shared.state.nextjob()
        p.is_refiner_pass = False
        timer.process.record('refine')
    return output


def process_decode(p: processing.StableDiffusionProcessing, output):
    if output is not None:
        if not hasattr(output, 'images') and hasattr(output, 'frames'):
            shared.log.debug(f'Generated: frames={len(output.frames[0])}')
            output.images = output.frames[0]
        model = shared.sd_model if not is_refiner_enabled(p) else shared.sd_refiner
        if not hasattr(model, 'vae'):
            if hasattr(model, 'pipe') and hasattr(model.pipe, 'vae'):
                model = model.pipe
        if (hasattr(model, "vae") or hasattr(model, "vqgan")) and output.images is not None and len(output.images) > 0:
            if p.hr_resize_mode > 0 and (p.hr_upscaler != 'None' or p.hr_resize_mode == 5):
                width = max(getattr(p, 'width', 0), getattr(p, 'hr_upscale_to_x', 0))
                height = max(getattr(p, 'height', 0), getattr(p, 'hr_upscale_to_y', 0))
            else:
                width = getattr(p, 'width', 0)
                height = getattr(p, 'height', 0)
            results = processing_vae.vae_decode(
                latents = output.images,
                model = model,
                full_quality = p.full_quality,
                width = width,
                height = height,
            )
        elif hasattr(output, 'images'):
            results = output.images
        else:
            shared.log.warning('Processing returned no results')
            results = []
    else:
        shared.log.warning('Processing returned no results')
        results = []
    return results


def update_pipeline(sd_model, p: processing.StableDiffusionProcessing):
    if sd_models.get_diffusers_task(sd_model) == sd_models.DiffusersTaskType.INPAINTING and getattr(p, 'image_mask', None) is None and p.task_args.get('image_mask', None) is None and getattr(p, 'mask', None) is None:
        shared.log.warning('Processing: mode=inpaint mask=None')
        sd_model = sd_models.set_diffuser_pipe(sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
    if shared.opts.cuda_compile_backend == "olive-ai":
        sd_model = olive_check_parameters_changed(p, is_refiner_enabled(p))
    if sd_model.__class__.__name__ == "OnnxRawPipeline":
        sd_model = preprocess_onnx_pipeline(p)
        global orig_pipeline # pylint: disable=global-statement
        orig_pipeline = sd_model # processed ONNX pipeline should not be replaced with original pipeline.
    if getattr(sd_model, "current_attn_name", None) != shared.opts.cross_attention_optimization:
        shared.log.info(f"Setting attention optimization: {shared.opts.cross_attention_optimization}")
        sd_models.set_diffusers_attention(sd_model)
    return sd_model


def process_diffusers(p: processing.StableDiffusionProcessing):
    debug(f'Process diffusers args: {vars(p)}')
    results = []
    p = restore_state(p)
    global orig_pipeline # pylint: disable=global-statement
    orig_pipeline = shared.sd_model

    if shared.state.interrupted or shared.state.skipped:
        shared.sd_model = orig_pipeline
        return results

    # sanitize init_images
    if hasattr(p, 'init_images') and getattr(p, 'init_images', None) is None:
        del p.init_images
    if hasattr(p, 'init_images') and not isinstance(getattr(p, 'init_images', []), list):
        p.init_images = [p.init_images]
    if len(getattr(p, 'init_images', [])) > 0:
        while len(p.init_images) < len(p.prompts):
            p.init_images.append(p.init_images[-1])
    # pipeline type is set earlier in processing, but check for sanity
    is_control = getattr(p, 'is_control', False) is True
    has_images = len(getattr(p, 'init_images' ,[])) > 0
    if sd_models.get_diffusers_task(shared.sd_model) != sd_models.DiffusersTaskType.TEXT_2_IMAGE and not has_images and not is_control:
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE) # reset pipeline
    if hasattr(shared.sd_model, 'unet') and hasattr(shared.sd_model.unet, 'config') and hasattr(shared.sd_model.unet.config, 'in_channels') and shared.sd_model.unet.config.in_channels == 9 and not is_control:
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.INPAINTING) # force pipeline
        if len(getattr(p, 'init_images', [])) == 0:
            p.init_images = [TF.to_pil_image(torch.rand((3, getattr(p, 'height', 512), getattr(p, 'width', 512))))]
    if p.prompts is None or len(p.prompts) == 0:
        p.prompts = p.all_prompts[p.iteration * p.batch_size:(p.iteration+1) * p.batch_size]
    if p.negative_prompts is None or len(p.negative_prompts) == 0:
        p.negative_prompts = p.all_negative_prompts[p.iteration * p.batch_size:(p.iteration+1) * p.batch_size]

    sd_models.move_model(shared.sd_model, devices.device)
    sd_models_compile.openvino_recompile_model(p, hires=False, refiner=False) # recompile if a parameter changes

    if 'base' not in p.skip:
        output = process_base(p)
    else:
        images, _index=shared.history.selected
        output = SimpleNamespace(images=images)

    if shared.state.interrupted or shared.state.skipped:
        shared.sd_model = orig_pipeline
        return results

    if 'hires' not in p.skip:
        output = process_hires(p, output)
        if shared.state.interrupted or shared.state.skipped:
            shared.sd_model = orig_pipeline
            return results

    if 'refine' not in p.skip:
        output = process_refine(p, output)
        if shared.state.interrupted or shared.state.skipped:
            shared.sd_model = orig_pipeline
            return results

    results = process_decode(p, output)

    timer.process.record('decode')
    shared.sd_model = orig_pipeline
    if p.state == '':
        global last_p # pylint: disable=global-statement
        last_p = p
    return results
