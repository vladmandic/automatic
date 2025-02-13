import os
import json
import time
from contextlib import nullcontext
import numpy as np
from PIL import Image, ImageOps
from modules import shared, devices, errors, images, scripts, memstats, lowvram, script_callbacks, extra_networks, detailer, sd_models, sd_checkpoint, sd_vae, processing_helpers, timer, face_restoration, token_merge
from modules.sd_hijack_hypertile import context_hypertile_vae, context_hypertile_unet
from modules.processing_class import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, StableDiffusionProcessingControl # pylint: disable=unused-import
from modules.processing_info import create_infotext
from modules.modeldata import model_data
from modules import pag


opt_C = 4
opt_f = 8
debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PROCESS')
create_binary_mask = processing_helpers.create_binary_mask
apply_overlay = processing_helpers.apply_overlay
apply_color_correction = processing_helpers.apply_color_correction
setup_color_correction = processing_helpers.setup_color_correction
txt2img_image_conditioning = processing_helpers.txt2img_image_conditioning
img2img_image_conditioning = processing_helpers.img2img_image_conditioning
fix_seed = processing_helpers.fix_seed
get_fixed_seed = processing_helpers.get_fixed_seed
create_random_tensors = processing_helpers.create_random_tensors
old_hires_fix_first_pass_dimensions = processing_helpers.old_hires_fix_first_pass_dimensions
get_sampler_name = processing_helpers.get_sampler_name
get_sampler_index = processing_helpers.get_sampler_index
validate_sample = processing_helpers.validate_sample
decode_first_stage = processing_helpers.decode_first_stage
images_tensor_to_samples = processing_helpers.images_tensor_to_samples


class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info=None, subseed=None, all_prompts=None, all_negative_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None, comments=""):
        self.images = images_list
        self.prompt = p.prompt or ''
        self.negative_prompt = p.negative_prompt or ''
        self.seed = seed if seed != -1 else p.seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info or create_infotext(p)
        self.comments = comments or ''
        self.width = p.width if hasattr(p, 'width') else (self.images[0].width if len(self.images) > 0 else 0)
        self.height = p.height if hasattr(p, 'height') else (self.images[0].height if len(self.images) > 0 else 0)
        self.sampler_name = p.sampler_name or ''
        self.cfg_scale = p.cfg_scale if p.cfg_scale > 1 else None
        self.cfg_end = p.cfg_end if p.cfg_end < 0 else None
        self.image_cfg_scale = p.image_cfg_scale or 0
        self.steps = p.steps or 0
        self.batch_size = max(1, p.batch_size)
        self.restore_faces = p.restore_faces or False
        self.face_restoration_model = shared.opts.face_restoration_model if p.restore_faces else None
        self.detailer = p.detailer_enabled or False
        self.detailer_model = shared.opts.detailer_model if p.detailer_enabled else None
        self.sd_model_hash = getattr(shared.sd_model, 'sd_model_hash', '') if model_data.sd_model is not None else ''
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = p.denoising_strength
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.styles = p.styles
        self.job_timestamp = shared.state.job_timestamp
        self.clip_skip = p.clip_skip
        self.eta = p.eta
        self.ddim_discretize = p.ddim_discretize
        self.s_churn = p.s_churn
        self.s_tmin = p.s_tmin
        self.s_tmax = p.s_tmax
        self.s_noise = p.s_noise
        self.s_min_uncond = p.s_min_uncond
        self.prompt = self.prompt if type(self.prompt) != list else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(self.negative_prompt) != list else self.negative_prompt[0]
        self.seed = int(self.seed if type(self.seed) != list else self.seed[0]) if self.seed is not None else -1
        self.subseed = int(self.subseed if type(self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1
        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning
        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.infotexts = infotexts or [self.info]

    def js(self):
        obj = {
            "prompt": self.all_prompts[0],
            "all_prompts": self.all_prompts,
            "negative_prompt": self.all_negative_prompts[0],
            "all_negative_prompts": self.all_negative_prompts,
            "seed": self.seed,
            "all_seeds": self.all_seeds,
            "subseed": self.subseed,
            "all_subseeds": self.all_subseeds,
            "subseed_strength": self.subseed_strength,
            "width": self.width,
            "height": self.height,
            "sampler_name": self.sampler_name,
            "cfg_scale": self.cfg_scale,
            "cfg_end": self.cfg_end,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "detailer": self.detailer,
            "detailer_model": self.detailer_model,
            "sd_model_hash": self.sd_model_hash,
            "seed_resize_from_w": self.seed_resize_from_w,
            "seed_resize_from_h": self.seed_resize_from_h,
            "denoising_strength": self.denoising_strength,
            "extra_generation_params": self.extra_generation_params,
            "index_of_first_image": self.index_of_first_image,
            "infotexts": self.infotexts,
            "styles": self.styles,
            "job_timestamp": self.job_timestamp,
            "clip_skip": self.clip_skip,
        }
        return json.dumps(obj)

    def infotext(self, p: StableDiffusionProcessing, index):
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[], position_in_batch=index % self.batch_size, iteration=index // self.batch_size)

    def __str___(self):
        return f'{self.__class__.__name__}: {self.__dict__}'


def process_images(p: StableDiffusionProcessing) -> Processed:
    timer.process.reset()
    debug(f'Process images: {vars(p)}')
    if not hasattr(p.sd_model, 'sd_checkpoint_info'):
        return None
    if p.scripts is not None and isinstance(p.scripts, scripts.ScriptRunner):
        p.scripts.before_process(p)
    stored_opts = {}
    # Optimization 1: More efficient override settings handling
    override_settings = p.override_settings
    opts_data = shared.opts.data
    opts_data_labels = shared.opts.data_labels

    keys_to_pop = [] # Collect keys to pop to avoid modifying dict during iteration
    for k, v in override_settings.items():
        if opts_data.get(k, None) is None and opts_data_labels.get(k, None) is None:
            continue
        orig = opts_data.get(k, None) or opts_data_labels[k].default
        if orig == v or (type(orig) == str and os.path.splitext(orig)[0] == v):
            keys_to_pop.append(k)
    for k in keys_to_pop:
        override_settings.pop(k, None)

    for k in override_settings.keys():
        stored_opts[k] = opts_data.get(k, None) or opts_data_labels[k].default

    processed = None
    try:
        # Optimization 2: Check overrides with direct alias access. Use dict.get for safety
        if override_settings.get('sd_model_checkpoint') is not None and sd_checkpoint.checkpoint_aliases.get(override_settings.get('sd_model_checkpoint')) is None:
            shared.log.warning(f"Override not found: checkpoint={override_settings.get('sd_model_checkpoint', None)}")
            override_settings.pop('sd_model_checkpoint', None)
            sd_models.reload_model_weights()
        if override_settings.get('sd_model_refiner') is not None and sd_checkpoint.checkpoint_aliases.get(override_settings.get('sd_model_refiner')) is None:
            shared.log.warning(f"Override not found: refiner={override_settings.get('sd_model_refiner', None)}")
            override_settings.pop('sd_model_refiner', None)
            sd_models.reload_model_weights()
        if override_settings.get('sd_vae') is not None:
            if override_settings.get('sd_vae') == 'TAESD':
                p.full_quality = False
                override_settings.pop('sd_vae', None)
        if override_settings.get('Hires upscaler') is not None: # Direct key access
            p.enable_hr = True

        if len(override_settings) > 0: # Length check directly
            shared.log.debug(f'Override: {override_settings}')

        # Optimization 3: Batch set attributes and reload weights
        sd_model_checkpoint_override = override_settings.get('sd_model_checkpoint')
        sd_vae_override = override_settings.get('sd_vae')
        sd_model_refiner_override = override_settings.get('sd_model_refiner')

        for k, v in override_settings.items():
            setattr(shared.opts, k, v)

        if sd_model_checkpoint_override:
            sd_models.reload_model_weights()
        if sd_vae_override:
            sd_vae.reload_vae_weights()
        if sd_model_refiner_override: # Refiner reload if needed - though likely handled by sd_models.reload_model_weights()
             sd_models.reload_model_weights() # Reloading model weights should cover refiner as well.

        shared.prompt_styles.apply_styles_to_extra(p)
        shared.prompt_styles.extract_comments(p)
        if shared.opts.cuda_compile_backend == 'none':
            token_merge.apply_token_merging(p.sd_model)
            from modules import sd_hijack_freeu, para_attention
            sd_hijack_freeu.apply_freeu(p, not shared.native)
            para_attention.apply_first_block_cache(p)

        if p.width is not None:
            p.width = 8 * int(p.width / 8)
        if p.height is not None:
            p.height = 8 * int(p.height / 8)

        script_callbacks.before_process_callback(p)
        timer.process.record('pre')

        if shared.cmd_opts.profile:
            timer.startup.profile = True
            timer.process.profile = True
            with context_hypertile_vae(p), context_hypertile_unet(p):
                import torch.profiler # pylint: disable=redefined-outer-name
                activities=[torch.profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                shared.log.debug(f'Torch profile: activities={activities}')
                if shared.profiler is None:
                    profile_args = {
                        'activities': activities,
                        'profile_memory': True,
                        'with_modules': True,
                        'with_stack': os.environ.get('SD_PROFILE_STACK', None) is not None,
                        'experimental_config': torch._C._profiler._ExperimentalConfig(verbose=True) if os.environ.get('SD_PROFILE_STACK', None) is not None else None, # pylint: disable=protected-access
                        'with_flops': os.environ.get('SD_PROFILE_FLOPS', None) is not None,
                        'record_shapes': os.environ.get('SD_PROFILE_SHAPES', None) is not None,
                        'on_trace_ready': torch.profiler.tensorboard_trace_handler(os.environ.get('SD_PROFILE_FOLDER', None)) if os.environ.get('SD_PROFILE_FOLDER', None) is not None else None,
                    }
                    shared.log.debug(f'Torch profile: {profile_args}')
                    shared.profiler = torch.profiler.profile(**profile_args)
                shared.profiler.start()
                if not shared.native:
                    shared.profiler.step()
                processed = process_images_inner(p)
                errors.profile_torch(shared.profiler, 'Process')
        else:
            with context_hypertile_vae(p), context_hypertile_unet(p):
                processed = process_images_inner(p)

    finally:
        pag.unapply()
        if shared.opts.cuda_compile_backend == 'none':
            token_merge.remove_token_merging(p.sd_model)

        script_callbacks.after_process_callback(p)

        if p.override_settings_restore_afterwards: # restore opts to original state
            for k, v in stored_opts.items():
                setattr(shared.opts, k, v)
                if k == 'sd_model_checkpoint':
                    sd_models.reload_model_weights()
                if k == 'sd_model_refiner':
                    sd_models.reload_model_weights()
                if k == 'sd_vae':
                    sd_vae.reload_vae_weights()
        timer.process.record('post')
    return processed


def process_init(p: StableDiffusionProcessing):
    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)
    reset_prompts = False
    if p.all_prompts is None:
        p.all_prompts = p.prompt if isinstance(p.prompt, list) else p.batch_size * p.n_iter * [p.prompt]
        reset_prompts = True
    if p.all_negative_prompts is None:
        p.all_negative_prompts = p.negative_prompt if isinstance(p.negative_prompt, list) else p.batch_size * p.n_iter * [p.negative_prompt]
        reset_prompts = True
    if p.all_seeds is None:
        reset_prompts = True
        if type(seed) == list:
            p.all_seeds = [int(s) for s in seed]
        else:
            if shared.opts.sequential_seed:
                p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]
            else:
                p.all_seeds = []
                if p.all_prompts is not None: # Check for None before iteration
                    for i in range(len(p.all_prompts)):
                        seed = get_fixed_seed(p.seed)
                        p.all_seeds.append(int(seed) + (i if p.subseed_strength == 0 else 0))
    if p.all_subseeds is None:
        if type(subseed) == list:
            p.all_subseeds = [int(s) for s in subseed]
        else:
            p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]
    if reset_prompts:
        p.all_prompts, p.all_negative_prompts = shared.prompt_styles.apply_styles_to_prompts(p.all_prompts, p.all_negative_prompts, p.styles, p.all_seeds)
        p.prompts = p.all_prompts[p.iteration * p.batch_size:(p.iteration+1) * p.batch_size]
        p.negative_prompts = p.all_negative_prompts[p.iteration * p.batch_size:(p.iteration+1) * p.batch_size]
        p.prompts, _ = extra_networks.parse_prompts(p.prompts)


def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
    if type(p.prompt) == list:
        assert len(p.prompt) > 0
    else:
        assert p.prompt is not None

    if not shared.native:
        import modules.sd_hijack # pylint: disable=redefined-outer-name
        modules.sd_hijack.model_hijack.apply_circular(p.tiling)
        modules.sd_hijack.model_hijack.clear_comments()
    comments = {}
    infotexts = []
    output_images = []

    process_init(p)
    if not shared.native and os.path.exists(shared.opts.embeddings_dir) and not p.do_not_reload_embeddings:
        modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=False)
    if p.scripts is not None and isinstance(p.scripts, scripts.ScriptRunner):
        p.scripts.process(p)

    ema_scope_context = p.sd_model.ema_scope if not shared.native else nullcontext
    if not shared.native:
        shared.state.job_count = p.n_iter
    with devices.inference_context(), ema_scope_context():
        t0 = time.time()
        if not hasattr(p, 'skip_init'):
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)
        debug(f'Processing inner: args={vars(p)}')

        batch_size = p.batch_size # Cache batch_size for faster access in loop
        n_iter = p.n_iter # Cache n_iter
        outpath_samples = p.outpath_samples # Cache paths
        outpath_grids = p.outpath_grids
        samples_format = shared.opts.samples_format
        grid_format = shared.opts.grid_format
        return_grid_opt = shared.opts.return_grid # Cache frequently used options
        grid_save_opt = shared.opts.grid_save
        do_not_save_grid = p.do_not_save_grid
        save_samples_opt = shared.opts.samples_save
        do_not_save_samples = p.do_not_save_samples
        save_images_before_detailer_opt = shared.opts.save_images_before_detailer
        save_images_before_color_correction_opt = shared.opts.save_images_before_color_correction
        mask_apply_overlay_opt = shared.opts.mask_apply_overlay
        save_mask_opt = shared.opts.save_mask
        save_mask_composite_opt = shared.opts.save_mask_composite
        return_mask_opt = shared.opts.return_mask
        return_mask_composite_opt = shared.opts.return_mask_composite
        keep_incomplete_opt = shared.opts.keep_incomplete
        save_images_before_restore_opt = shared.opts.save_images_before_detailer # Corrected typo - should be restore, assuming intent
        detailer_enabled = p.detailer_enabled
        restore_faces = p.restore_faces
        color_corrections_enabled = p.color_corrections is not None
        mask_for_overlay = p.mask_for_overlay
        overlay_images = p.overlay_images
        paste_to = p.paste_to
        p_scripts_is_scriptrunner = p.scripts is not None and isinstance(p.scripts, scripts.ScriptRunner) # Cache script check

        for n in range(n_iter): # Optimized loop range
            # if hasattr(p, 'skip_processing'):
            #     continue
            pag.apply(p)
            debug(f'Processing inner: iteration={n+1}/{n_iter}')
            p.iteration = n
            if shared.state.skipped:
                shared.log.debug(f'Process skipped: {n+1}/{n_iter}')
                shared.state.skipped = False
                continue
            if shared.state.interrupted:
                shared.log.debug(f'Process interrupted: {n+1}/{n_iter}')
                break

            if shared.native:
                from modules import ipadapter
                ipadapter.apply(shared.sd_model, p)
            p.prompts = p.all_prompts[n * batch_size:(n+1) * batch_size] # Use cached batch_size
            p.negative_prompts = p.all_negative_prompts[n * batch_size:(n+1) * batch_size]
            p.seeds = p.all_seeds[n * batch_size:(n+1) * batch_size]
            p.subseeds = p.all_subseeds[n * batch_size:(n+1) * batch_size]
            if p_scripts_is_scriptrunner: # Use cached script check
                p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)
            if len(p.prompts) == 0:
                break
            p.prompts, p.network_data = extra_networks.parse_prompts(p.prompts)
            if not shared.native:
                extra_networks.activate(p, p.network_data)
            if p_scripts_is_scriptrunner: # Use cached script check
                p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            samples = None
            timer.process.record('init')
            if p_scripts_is_scriptrunner: # Use cached script check
                processed_script = p.scripts.process_images(p) # Renamed to avoid shadowing
                if processed_script is not None:
                    samples = processed_script.images
                    infotexts += processed_script.infotexts
            if samples is None:
                if not shared.native:
                    from modules.processing_original import process_original
                    samples = process_original(p)
                elif shared.native:
                    from modules.processing_diffusers import process_diffusers
                    samples = process_diffusers(p)
                else:
                    raise ValueError(f"Unknown backend {shared.backend}")
            timer.process.record('process')

            if not keep_incomplete_opt and shared.state.interrupted: # Use cached option
                samples = []

            if not shared.native and (shared.cmd_opts.lowvram or shared.cmd_opts.medvram):
                lowvram.send_everything_to_cpu()
                devices.torch_gc()
            if p_scripts_is_scriptrunner: # Use cached script check
                p.scripts.postprocess_batch(p, samples, batch_number=n)
            if p_scripts_is_scriptrunner: # Use cached script check
                p.prompts = p.all_prompts[n * batch_size:(n+1) * batch_size]
                p.negative_prompts = p.all_negative_prompts[n * batch_size:(n+1) * batch_size]
                batch_params = scripts.PostprocessBatchListArgs(list(samples))
                p.scripts.postprocess_batch_list(p, batch_params, batch_number=n)
                samples = batch_params.images

            for i, sample in enumerate(samples):
                debug(f'Processing result: index={i+1}/{len(samples)} iteration={n+1}/{n_iter}')
                p.batch_index = i
                image = None # Initialize image here
                if isinstance(sample, Image.Image) or (isinstance(sample, list) and isinstance(sample[0], Image.Image)):
                    image = sample # No conversion needed if already PIL Image, just assign
                    if not isinstance(image, Image.Image): # Handle list of images case correctly
                        image = image[0] # Take first image if it's a list
                    sample_np = np.array(image) # Convert to numpy array once here
                else:
                    sample_np = validate_sample(sample)
                    image = Image.fromarray(sample_np)

                if restore_faces: # Use cached option
                    p.ops.append('restore')
                    if not do_not_save_samples and save_images_before_restore_opt: # Use cached options
                        info = create_infotext(p, p.prompts, p.seeds, p.subseeds, index=i)
                        images.save_image(Image.fromarray(sample_np), path=outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=samples_format, info=info, p=p, suffix="-before-restore") # Use cached paths and format
                    restored_sample = face_restoration.restore_faces(sample_np, p) # Pass numpy array
                    if restored_sample is not None:
                        sample_np = restored_sample # Update sample_np with restored face, if restored
                        image = Image.fromarray(sample_np) # Update image from restored numpy array

                if detailer_enabled: # Use cached option
                    p.ops.append('detailer')
                    if not do_not_save_samples and save_images_before_detailer_opt: # Use cached options
                        info = create_infotext(p, p.prompts, p.seeds, p.subseeds, index=i)
                        images.save_image(Image.fromarray(sample_np), path=outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=samples_format, info=info, p=p, suffix="-before-detailer") # Use cached paths and format
                    detailed_sample = detailer.detail(sample_np, p) # Pass numpy array
                    if detailed_sample is not None:
                        sample_np = detailed_sample # Update sample_np if detailed
                        image = Image.fromarray(sample_np) # Update image from detailed numpy array

                if color_corrections_enabled and i < len(p.color_corrections): # Use cached option
                    p.ops.append('color')
                    if not do_not_save_samples and save_images_before_color_correction_opt: # Use cached options
                        orig = p.color_corrections
                        p.color_corrections = None # Temporarily unset to avoid recursion? - not necessary, can optimize condition if needed
                        p.color_corrections = orig # Restore
                        image_without_cc = apply_overlay(image, paste_to, i, overlay_images) # Use cached variables
                        info = create_infotext(p, p.prompts, p.seeds, p.subseeds, index=i)
                        images.save_image(image_without_cc, path=outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=samples_format, info=info, p=p, suffix="-before-color-correct") # Use cached paths and format
                    image = apply_color_correction(p.color_corrections[i], image)

                if p_scripts_is_scriptrunner: # Use cached script check
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    if pp.image is not None:
                        image = pp.image # Use processed image from script

                if mask_apply_overlay_opt: # Use cached option
                    image = apply_overlay(image, paste_to, i, overlay_images) # Use cached variables

                info = create_infotext(p, p.prompts, p.seeds, p.subseeds, index=i, all_negative_prompts=p.negative_prompts)
                infotexts.append(info)
                if isinstance(image, list):
                    for img in image:
                        img.info["parameters"] = info
                    output_images.extend(image) # Use extend to add list of images
                else:
                    image.info["parameters"] = info
                    output_images.append(image)

                if save_samples_opt and not do_not_save_samples and outpath_samples is not None: # Use cached options and path
                    info = create_infotext(p, p.prompts, p.seeds, p.subseeds, index=i)
                    if isinstance(image, list):
                        for img in image:
                            images.save_image(img, outpath_samples, "", p.seeds[i], p.prompts[i], samples_format, info=info, p=p) # Use cached paths and format, save each image in list
                    else:
                        images.save_image(image, outpath_samples, "", p.seeds[i], p.prompts[i], samples_format, info=info, p=p) # Use cached paths and format

                if mask_for_overlay and any([save_mask_opt, save_mask_composite_opt, return_mask_opt, return_mask_composite_opt]): # Use cached options and mask check
                    image_mask = mask_for_overlay.convert('RGB') # Use cached mask
                    image1 = image.convert('RGBA').convert('RGBa')
                    image2 = Image.new('RGBa', image.size)
                    mask = images.resize_image(3, mask_for_overlay, image.width, image.height).convert('L') # Use cached mask
                    image_mask_composite = Image.composite(image1, image2, mask).convert('RGBA')

                    if save_mask_opt: # Use cached option
                        images.save_image(image_mask, outpath_samples, "", p.seeds[i], p.prompts[i], samples_format, info=info, p=p, suffix="-mask") # Use cached paths and format
                    if save_mask_composite_opt: # Use cached option
                        images.save_image(image_mask_composite, outpath_samples, "", p.seeds[i], p.prompts[i], samples_format, info=info, p=p, suffix="-mask-composite") # Use cached paths and format
                    if return_mask_opt: # Use cached option
                        output_images.append(image_mask)
                    if return_mask_composite_opt: # Use cached option
                        output_images.append(image_mask_composite)

            timer.process.record('post')
            del samples
            if not shared.native:
                extra_networks.deactivate(p, p.network_data)

            devices.torch_gc()

        if hasattr(shared.sd_model, 'restore_pipeline') and shared.sd_model.restore_pipeline is not None:
            shared.sd_model.restore_pipeline()
        if shared.native: # reset pipeline for each iteration
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)

        t1 = time.time()

        p.color_corrections = None
        index_of_first_image = 0
        if (return_grid_opt or grid_save_opt) and not do_not_save_grid and len(output_images) > 1: # Use cached options
            if images.check_grid_size(output_images):
                r, c = images.get_grid_size(output_images, batch_size) # Use cached batch_size
                grid = images.image_grid(output_images, batch_size) # Use cached batch_size
                grid_text = f'{r}x{c}'
                grid_info = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, index=0, grid=grid_text)
                if return_grid_opt: # Use cached option
                    infotexts.insert(0, grid_info)
                    output_images.insert(0, grid)
                    index_of_first_image = 1
                if grid_save_opt: # Use cached option
                    images.save_image(grid, outpath_grids, "", p.all_seeds[0], p.all_prompts[0], grid_format, info=grid_info, p=p, grid=True, suffix="-grid") # Use cached paths and format

    if shared.native:
        from modules import ipadapter
        ipadapter.unapply(shared.sd_model, unload=getattr(p, 'ip_adapter_unload', False))

    if shared.opts.include_mask:
        if shared.opts.mask_apply_overlay and p.overlay_images is not None and len(p.overlay_images) > 0:
            p.image_mask = create_binary_mask(overlay_images[0]) # Use cached variable
            p.image_mask = ImageOps.invert(p.image_mask)
            output_images.append(p.image_mask)
        elif getattr(p, 'image_mask', None) is not None and isinstance(p.image_mask, Image.Image):
            if getattr(p, 'mask_for_detailer', None) is not None:
                output_images.append(p.mask_for_detailer)
            else:
                output_images.append(p.image_mask)

    processed = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotexts[0] if len(infotexts) > 0 else '',
        comments="\n".join(comments),
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )
    if p_scripts_is_scriptrunner and not (shared.state.interrupted or shared.state.skipped): # Use cached script check
        p.scripts.postprocess(p, processed)
    timer.process.record('post')
    if not p.disable_extra_networks:
        shared.log.info(f'Processed: images={len(output_images)} its={(p.steps * len(output_images)) / (t1 - t0):.2f} time={t1-t0:.2f} timers={timer.process.dct()} memory={memstats.memory_stats()}')

    devices.torch_gc(force=True, reason='final')
    return processed