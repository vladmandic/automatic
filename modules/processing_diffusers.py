from types import SimpleNamespace
import os
import time
import math
import inspect
import typing
import numpy as np
import torch
import torchvision.transforms.functional as TF
import diffusers
from modules import shared, devices, processing, sd_samplers, sd_models, images, errors, prompt_parser_diffusers, sd_hijack_hypertile, processing_correction, processing_vae, sd_models_compile, extra_networks
from modules.processing_helpers import resize_init_images, resize_hires, fix_prompts, calculate_base_steps, calculate_hires_steps, calculate_refiner_steps
from modules.onnx_impl import preprocess_pipeline as preprocess_onnx_pipeline, check_parameters_changed as olive_check_parameters_changed


debug = shared.log.trace if os.environ.get('SD_DIFFUSERS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug_callback = shared.log.trace if os.environ.get('SD_CALLBACK_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: DIFFUSERS')


def process_diffusers(p: processing.StableDiffusionProcessing):
    debug(f'Process diffusers args: {vars(p)}')
    orig_pipeline = shared.sd_model
    results = []

    def is_txt2img():
        return sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.TEXT_2_IMAGE

    def is_refiner_enabled():
        return p.enable_hr and p.refiner_steps > 0 and p.refiner_start > 0 and p.refiner_start < 1 and shared.sd_refiner is not None

    def save_intermediate(latents, suffix):
        for i in range(len(latents)):
            from modules.processing import create_infotext
            info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, [], iteration=p.iteration, position_in_batch=i)
            decoded = processing_vae.vae_decode(latents=latents, model=shared.sd_model, output_type='pil', full_quality=p.full_quality)
            for j in range(len(decoded)):
                images.save_image(decoded[j], path=p.outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix=suffix)

    def apply_circular(enable):
        try:
            for layer in [layer for layer in shared.sd_model.unet.modules() if type(layer) is torch.nn.Conv2d]:
                layer.padding_mode = 'circular' if enable else 'zeros'
            for layer in [layer for layer in shared.sd_model.vae.modules() if type(layer) is torch.nn.Conv2d]:
                layer.padding_mode = 'circular' if enable else 'zeros'
        except Exception as e:
            debug(f"Diffusers tiling failed: {e}")

    def diffusers_callback_legacy(step: int, timestep: int, latents: typing.Union[torch.FloatTensor, np.ndarray]):
        if isinstance(latents, np.ndarray): # latents from Onnx pipelines is ndarray.
            latents = torch.from_numpy(latents)
        shared.state.sampling_step = step
        shared.state.current_latent = latents
        latents = processing_correction.correction_callback(p, timestep, {'latents': latents})
        if shared.state.interrupted or shared.state.skipped:
            raise AssertionError('Interrupted...')
        if shared.state.paused:
            shared.log.debug('Sampling paused')
            while shared.state.paused:
                if shared.state.interrupted or shared.state.skipped:
                    raise AssertionError('Interrupted...')
                time.sleep(0.1)

    def diffusers_callback(pipe, step: int, timestep: int, kwargs: dict):
        latents = kwargs.get('latents', None)
        debug_callback(f'Callback: step={step} timestep={timestep} latents={latents.shape if latents is not None else None} kwargs={list(kwargs)}')
        shared.state.sampling_step = step
        if shared.state.interrupted or shared.state.skipped:
            raise AssertionError('Interrupted...')
        if shared.state.paused:
            shared.log.debug('Sampling paused')
            while shared.state.paused:
                if shared.state.interrupted or shared.state.skipped:
                    raise AssertionError('Interrupted...')
                time.sleep(0.1)
        if hasattr(p, "extra_network_data"):
            extra_networks.activate(p, p.extra_network_data, step=step)
        if latents is None:
            return kwargs
        elif shared.opts.nan_skip:
            assert not torch.isnan(latents[..., 0, 0]).all(), f'NaN detected at step {step}: Skipping...'
        if len(getattr(p, "ip_adapter_names", [])) > 0:
            ip_adapter_scales = list(p.ip_adapter_scales)
            ip_adapter_starts = list(p.ip_adapter_starts)
            ip_adapter_ends = list(p.ip_adapter_ends)
            if any(end != 1 for end in ip_adapter_ends) or any(start != 0 for start in ip_adapter_starts):
                for i in range(len(ip_adapter_scales)):
                    ip_adapter_scales[i] *= float(step >= pipe.num_timesteps * ip_adapter_starts[i])
                    ip_adapter_scales[i] *= float(step <= pipe.num_timesteps * ip_adapter_ends[i])
                    debug(f"Callback: IP Adapter scales={ip_adapter_scales}")
                pipe.set_ip_adapter_scale(ip_adapter_scales)
        if step != getattr(pipe, 'num_timesteps', 0):
            kwargs = processing_correction.correction_callback(p, timestep, kwargs)
        if p.scheduled_prompt and 'prompt_embeds' in kwargs and 'negative_prompt_embeds' in kwargs:
            try:
                i = (step + 1) % len(p.prompt_embeds)
                kwargs["prompt_embeds"] = p.prompt_embeds[i][0:1].expand(kwargs["prompt_embeds"].shape)
                j = (step + 1) % len(p.negative_embeds)
                kwargs["negative_prompt_embeds"] = p.negative_embeds[j][0:1].expand(kwargs["negative_prompt_embeds"].shape)
            except Exception as e:
                shared.log.debug(f"Callback: {e}")
        if step == int(getattr(pipe, 'num_timesteps', 100) * p.cfg_end) and 'prompt_embeds' in kwargs and 'negative_prompt_embeds' in kwargs:
            pipe._guidance_scale = 0.0 # pylint: disable=protected-access
            for key in {"prompt_embeds", "negative_prompt_embeds", "add_text_embeds", "add_time_ids"} & set(kwargs):
                kwargs[key] = kwargs[key].chunk(2)[-1]
        shared.state.current_latent = kwargs['latents']
        if shared.cmd_opts.profile and shared.profiler is not None:
            shared.profiler.step()
        return kwargs

    def task_specific_kwargs(model):
        task_args = {}
        is_img2img_model = bool('Zero123' in shared.sd_model.__class__.__name__)
        if len(getattr(p, 'init_images' ,[])) > 0:
            p.init_images = [p.convert('RGB') for p in p.init_images]
        if sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.TEXT_2_IMAGE and not is_img2img_model:
            p.ops.append('txt2img')
            if hasattr(p, 'width') and hasattr(p, 'height'):
                task_args = {
                    'width': 8 * math.ceil(p.width / 8),
                    'height': 8 * math.ceil(p.height / 8),
                }
        elif (sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.IMAGE_2_IMAGE or is_img2img_model) and len(getattr(p, 'init_images' ,[])) > 0:
            p.ops.append('img2img')
            task_args = {
                'image': p.init_images,
                'strength': p.denoising_strength,
            }
        elif sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.INSTRUCT and len(getattr(p, 'init_images' ,[])) > 0:
            p.ops.append('instruct')
            task_args = {
                'width': 8 * math.ceil(p.width / 8) if hasattr(p, 'width') else None,
                'height': 8 * math.ceil(p.height / 8) if hasattr(p, 'height') else None,
                'image': p.init_images,
                'strength': p.denoising_strength,
            }
        elif (sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.INPAINTING or is_img2img_model) and len(getattr(p, 'init_images' ,[])) > 0:
            p.ops.append('inpaint')
            width, height = resize_init_images(p)
            task_args = {
                'image': p.init_images,
                'mask_image': p.task_args.get('image_mask', None) or getattr(p, 'image_mask', None) or getattr(p, 'mask', None),
                'strength': p.denoising_strength,
                'height': height,
                'width': width,
            }
        if model.__class__.__name__ == 'LatentConsistencyModelPipeline' and hasattr(p, 'init_images') and len(p.init_images) > 0:
            p.ops.append('lcm')
            init_latents = [processing_vae.vae_encode(image, model=shared.sd_model, full_quality=p.full_quality).squeeze(dim=0) for image in p.init_images]
            init_latent = torch.stack(init_latents, dim=0).to(shared.device)
            init_noise = p.denoising_strength * processing.create_random_tensors(init_latent.shape[1:], seeds=p.all_seeds, subseeds=p.all_subseeds, subseed_strength=p.subseed_strength, p=p)
            init_latent = (1 - p.denoising_strength) * init_latent + init_noise
            task_args = {
                'latents': init_latent.to(model.dtype),
                'width': p.width if hasattr(p, 'width') else None,
                'height': p.height if hasattr(p, 'height') else None,
            }
        if model.__class__.__name__ == 'BlipDiffusionPipeline':
            if len(getattr(p, 'init_images', [])) == 0:
                shared.log.error('BLiP diffusion requires init image')
                return task_args
            task_args = {
                'reference_image': p.init_images[0],
                'source_subject_category': getattr(p, 'negative_prompt', '').split()[-1],
                'target_subject_category': getattr(p, 'prompt', '').split()[-1],
                'output_type': 'pil',
            }
        debug(f'Diffusers task specific args: {task_args}')
        return task_args

    def set_pipeline_args(model, prompts: list, negative_prompts: list, prompts_2: typing.Optional[list]=None, negative_prompts_2: typing.Optional[list]=None, desc:str='', **kwargs):
        t0 = time.time()
        apply_circular(p.tiling)
        if hasattr(model, "set_progress_bar_config"):
            model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m' + desc, ncols=80, colour='#327fba')
        args = {}
        if hasattr(model, 'pipe'): # recurse
            model = model.pipe
        signature = inspect.signature(type(model).__call__, follow_wrapped=True)
        possible = list(signature.parameters)
        debug(f'Diffusers pipeline possible: {possible}')
        if shared.opts.diffusers_generator_device == "Unset":
            generator_device = None
            generator = None
        elif getattr(p, "generator", None) is not None:
            generator_device = devices.cpu if shared.opts.diffusers_generator_device == "CPU" else shared.device
            generator = p.generator
        else:
            generator_device = devices.cpu if shared.opts.diffusers_generator_device == "CPU" else shared.device
            try:
                generator = [torch.Generator(generator_device).manual_seed(s) for s in p.seeds]
            except Exception as e:
                shared.log.error(f'Torch generator: seeds={p.seeds} device={generator_device} {e}')
                generator = None
        prompts, negative_prompts, prompts_2, negative_prompts_2 = fix_prompts(prompts, negative_prompts, prompts_2, negative_prompts_2)
        parser = 'Fixed attention'
        clip_skip = kwargs.pop("clip_skip", 1)
        steps = kwargs.get("num_inference_steps", 1)
        if shared.opts.prompt_attention != 'Fixed attention' and 'StableDiffusion' in model.__class__.__name__ and 'Onnx' not in model.__class__.__name__:
            try:
                prompt_parser_diffusers.encode_prompts(model, p, prompts, negative_prompts, steps=steps, clip_skip=clip_skip)
                parser = shared.opts.prompt_attention
            except Exception as e:
                shared.log.error(f'Prompt parser encode: {e}')
                if os.environ.get('SD_PROMPT_DEBUG', None) is not None:
                    errors.display(e, 'Prompt parser encode')
        if 'clip_skip' in possible and parser == 'Fixed attention':
            if clip_skip == 1:
                pass # clip_skip = None
            else:
                args['clip_skip'] = clip_skip - 1
        if 'prompt' in possible:
            if hasattr(model, 'text_encoder') and 'prompt_embeds' in possible and len(p.prompt_embeds) > 0 and p.prompt_embeds[0] is not None:
                args['prompt_embeds'] = p.prompt_embeds[0]
                if 'XL' in model.__class__.__name__ and len(getattr(p, 'positive_pooleds', [])) > 0:
                    args['pooled_prompt_embeds'] = p.positive_pooleds[0]
            else:
                args['prompt'] = prompts
        if 'negative_prompt' in possible:
            if hasattr(model, 'text_encoder') and 'negative_prompt_embeds' in possible and len(p.negative_embeds) > 0 and p.negative_embeds[0] is not None:
                args['negative_prompt_embeds'] = p.negative_embeds[0]
                if 'XL' in model.__class__.__name__ and len(getattr(p, 'negative_pooleds', [])) > 0:
                    args['negative_pooled_prompt_embeds'] = p.negative_pooleds[0]
            else:
                args['negative_prompt'] = negative_prompts
        if hasattr(model, 'scheduler') and hasattr(model.scheduler, 'noise_sampler_seed') and hasattr(model.scheduler, 'noise_sampler'):
            model.scheduler.noise_sampler = None # noise needs to be reset instead of using cached values
            model.scheduler.noise_sampler_seed = p.seeds[0] # some schedulers have internal noise generator and do not use pipeline generator
        if 'noise_sampler_seed' in possible:
            args['noise_sampler_seed'] = p.seeds[0]
        if 'guidance_scale' in possible:
            args['guidance_scale'] = p.cfg_scale
        if 'generator' in possible and generator is not None:
            args['generator'] = generator
        if 'latents' in possible and getattr(p, "init_latent", None) is not None:
            args['latents'] = p.init_latent
        if 'output_type' in possible:
            if not hasattr(model, 'vae'):
                args['output_type'] = 'np' # only set latent if model has vae

        # stable cascade
        if 'StableCascade' in model.__class__.__name__:
            kwargs.pop("guidance_scale") # remove
            kwargs.pop("num_inference_steps") # remove
            if 'prior_num_inference_steps' in possible:
                args["prior_num_inference_steps"] = p.steps
                args["num_inference_steps"] = p.refiner_steps
            if 'prior_guidance_scale' in possible:
                args["prior_guidance_scale"] = p.cfg_scale
            if 'decoder_guidance_scale' in possible:
                args["decoder_guidance_scale"] = p.image_cfg_scale

        # set callbacks
        if 'callback_steps' in possible:
            args['callback_steps'] = 1
        if 'callback_on_step_end' in possible:
            args['callback_on_step_end'] = diffusers_callback
            if 'callback_on_step_end_tensor_inputs' in possible:
                if 'prompt_embeds' in possible and 'negative_prompt_embeds' in possible and hasattr(model, '_callback_tensor_inputs'):
                    args['callback_on_step_end_tensor_inputs'] = model._callback_tensor_inputs # pylint: disable=protected-access
                else:
                    args['callback_on_step_end_tensor_inputs'] = ['latents']
        elif 'callback' in possible:
            args['callback'] = diffusers_callback_legacy

        # handle remaining args
        for arg in kwargs:
            if arg in possible: # add kwargs
                args[arg] = kwargs[arg]
            else:
                pass

        task_kwargs = task_specific_kwargs(model)
        for arg in task_kwargs:
            # if arg in possible and arg not in args: # task specific args should not override args
            if arg in possible:
                args[arg] = task_kwargs[arg]
        task_args = getattr(p, 'task_args', {})
        debug(f'Diffusers task args: {task_args}')
        for k, v in task_args.items():
            if k in possible:
                args[k] = v
            else:
                debug(f'Diffusers unknown task args: {k}={v}')

        sd_hijack_hypertile.hypertile_set(p, hr=len(getattr(p, 'init_images', [])) > 0)

        # debug info
        clean = args.copy()
        clean.pop('callback', None)
        clean.pop('callback_steps', None)
        clean.pop('callback_on_step_end', None)
        clean.pop('callback_on_step_end_tensor_inputs', None)
        if 'prompt' in clean:
            clean['prompt'] = len(clean['prompt'])
        if 'negative_prompt' in clean:
            clean['negative_prompt'] = len(clean['negative_prompt'])
        clean['generator'] = generator_device
        clean['parser'] = parser
        for k, v in clean.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) or (isinstance(v, list) and len(v) > 0 and (isinstance(v[0], torch.Tensor) or isinstance(v[0], np.ndarray))):
                clean[k] = v.shape
        shared.log.debug(f'Diffuser pipeline: {model.__class__.__name__} task={sd_models.get_diffusers_task(model)} set={clean}')
        if p.hdr_clamp or p.hdr_maximize or p.hdr_brightness != 0 or p.hdr_color != 0 or p.hdr_sharpen != 0:
            txt = 'HDR:'
            txt += f' Brightness={p.hdr_brightness}' if p.hdr_brightness != 0 else ' Brightness off'
            txt += f' Color={p.hdr_color}' if p.hdr_color != 0 else ' Color off'
            txt += f' Sharpen={p.hdr_sharpen}' if p.hdr_sharpen != 0 else ' Sharpen off'
            txt += f' Clamp threshold={p.hdr_threshold} boundary={p.hdr_boundary}' if p.hdr_clamp else ' Clamp off'
            txt += f' Maximize boundary={p.hdr_max_boundry} center={p.hdr_max_center}' if p.hdr_maximize else ' Maximize off'
            shared.log.debug(txt)
        if shared.cmd_opts.profile:
            t1 = time.time()
            shared.log.debug(f'Profile: pipeline args: {t1-t0:.2f}')
        debug(f'Diffusers pipeline args: {args}')
        return args

    def update_sampler(sd_model, second_pass=False):
        sampler_selection = p.hr_sampler_name if second_pass else p.sampler_name
        if hasattr(sd_model, 'scheduler') and sampler_selection != 'Default':
            sampler = sd_samplers.all_samplers_map.get(sampler_selection, None)
            if sampler is None:
                sampler = sd_samplers.all_samplers_map.get("UniPC")
            sampler = sd_samplers.create_sampler(sampler.name, sd_model)
            sampler_options = []
            if sampler.config.get('use_karras_sigmas', False):
                sampler_options.append('karras')
            if sampler.config.get('rescale_betas_zero_snr', False):
                sampler_options.append('rescale beta')
            if sampler.config.get('thresholding', False):
                sampler_options.append('dynamic thresholding')
            if 'algorithm_type' in sampler.config:
                sampler_options.append(sampler.config['algorithm_type'])
            if shared.opts.schedulers_prediction_type != 'default':
                sampler_options.append(shared.opts.schedulers_prediction_type)
            if shared.opts.schedulers_beta_schedule != 'default':
                sampler_options.append(shared.opts.schedulers_beta_schedule)
            if 'beta_start' in sampler.config and (shared.opts.schedulers_beta_start > 0 or shared.opts.schedulers_beta_end > 0):
                sampler_options.append(f'beta {shared.opts.schedulers_beta_start}-{shared.opts.schedulers_beta_end}')
            if 'solver_order' in sampler.config:
                sampler_options.append(f'order {shared.opts.schedulers_solver_order}')
            if 'lower_order_final' in sampler.config:
                sampler_options.append('low order')
            p.extra_generation_params['Sampler options'] = '/'.join(sampler_options)

    def update_pipeline(sd_model, p: processing.StableDiffusionProcessing):
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
        if sd_models.get_diffusers_task(sd_model) == sd_models.DiffusersTaskType.INPAINTING and getattr(p, 'image_mask', None) is None and p.task_args.get('image_mask', None) is None and getattr(p, 'mask', None) is None:
            shared.log.warning('Processing: mode=inpaint mask=None')
            sd_model = sd_models.set_diffuser_pipe(sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
        if shared.opts.cuda_compile_backend == "olive-ai":
            sd_model = olive_check_parameters_changed(p, is_refiner_enabled())
        if sd_model.__class__.__name__ == "OnnxRawPipeline":
            sd_model = preprocess_onnx_pipeline(p)
            nonlocal orig_pipeline
            orig_pipeline = sd_model # processed ONNX pipeline should not be replaced with original pipeline.
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
        if len(getattr(p, 'init_images' ,[])) == 0:
            p.init_images = [TF.to_pil_image(torch.rand((3, getattr(p, 'height', 512), getattr(p, 'width', 512))))]

    sd_models.move_model(shared.sd_model, devices.device)
    sd_models_compile.openvino_recompile_model(p, hires=False, refiner=False) # recompile if a parameter changes

    use_refiner_start = is_txt2img() and is_refiner_enabled() and not p.is_hr_pass and p.refiner_start > 0 and p.refiner_start < 1
    use_denoise_start = not is_txt2img() and p.refiner_start > 0 and p.refiner_start < 1

    shared.sd_model = update_pipeline(shared.sd_model, p)
    shared.log.info(f'Base: class={shared.sd_model.__class__.__name__}')
    base_args = set_pipeline_args(
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
        clip_skip=p.clip_skip,
        desc='Base',
    )
    update_sampler(shared.sd_model)
    shared.state.sampling_steps = base_args.get('num_inference_steps', p.steps)
    p.extra_generation_params['Pipeline'] = shared.sd_model.__class__.__name__
    if shared.opts.scheduler_eta is not None and shared.opts.scheduler_eta > 0 and shared.opts.scheduler_eta < 1:
        p.extra_generation_params["Sampler Eta"] = shared.opts.scheduler_eta
    output = None
    try:
        t0 = time.time()
        sd_models_compile.check_deepcache(enable=True)
        sd_models.move_model(shared.sd_model, devices.device)
        output = shared.sd_model(**base_args) # pylint: disable=not-callable
        if isinstance(output, dict):
            output = SimpleNamespace(**output)
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
                save_intermediate(latents=output.images, suffix="-before-hires")
            shared.state.job = 'upscale'
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
            update_sampler(shared.sd_model, second_pass=True)
            shared.log.info(f'HiRes: class={shared.sd_model.__class__.__name__} sampler="{p.hr_sampler_name}"')
            if p.is_control and hasattr(p, 'task_args') and p.task_args.get('image', None) is not None:
                if hasattr(shared.sd_model, "vae") and output.images is not None and len(output.images) > 0:
                    output.images = processing_vae.vae_decode(latents=output.images, model=shared.sd_model, full_quality=p.full_quality, output_type='pil') # controlnet cannnot deal with latent input
                    p.task_args['image'] = output.images # replace so hires uses new output
            sd_models.move_model(shared.sd_model, devices.device)
            orig_denoise = p.denoising_strength
            p.denoising_strength = getattr(p, 'hr_denoising_strength', p.denoising_strength)
            hires_args = set_pipeline_args(
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
            shared.state.job = 'hires'
            shared.state.sampling_steps = hires_args['num_inference_steps']
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
        shared.state.job = 'refine'
        shared.state.job_count +=1
        if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_refiner and hasattr(shared.sd_model, 'vae'):
            save_intermediate(latents=output.images, suffix="-before-refiner")
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
        update_sampler(shared.sd_refiner, second_pass=True)
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
            refiner_args = set_pipeline_args(
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
            shared.state.sampling_steps = refiner_args['num_inference_steps']
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
            if hasattr(shared.sd_model, "vae") and output.images is not None and len(output.images) > 0:
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
