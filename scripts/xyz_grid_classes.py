from scripts.xyz_grid_shared import apply_field, apply_task_args, apply_setting, apply_prompt, apply_order, apply_sampler, apply_hr_sampler_name, confirm_samplers, apply_checkpoint, apply_refiner, apply_unet, apply_dict, apply_clip_skip, apply_vae, list_lora, apply_lora, apply_lora_strength, apply_te, apply_styles, apply_upscaler, apply_context, apply_detailer, apply_override, apply_processing, apply_options, apply_seed, format_value_add_label, format_value, format_value_join_list, do_nothing, format_nothing, str_permutations # pylint: disable=no-name-in-module, unused-import
from modules import shared, shared_items, sd_samplers, ipadapter, sd_models, sd_vae, sd_unet


class AxisOption:
    def __init__(self, label, tipe, apply, fmt=format_value_add_label, confirm=None, cost=0.0, choices=None):
        self.label = label
        self.type = tipe
        self.apply = apply
        self.format_value = fmt
        self.confirm = confirm
        self.cost = cost
        self.choices = choices


class AxisOptionImg2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = True

class AxisOptionTxt2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = False


class SharedSettingsStackHelper(object):
    vae = None
    schedulers_solver_order = None
    tome_ratio = None
    todo_ratio = None
    sd_model_checkpoint = None
    sd_model_refiner = None
    sd_model_dict = None
    sd_vae = None
    sd_unet = None
    sd_text_encoder = None
    extra_networks_default_multiplier = None
    disable_weights_auto_swap = None
    prompt_attention = None

    def __enter__(self):
        #Save overridden settings so they can be restored later.
        self.vae = shared.opts.sd_vae
        self.schedulers_solver_order = shared.opts.schedulers_solver_order
        self.tome_ratio = shared.opts.tome_ratio
        self.todo_ratio = shared.opts.todo_ratio
        self.sd_model_checkpoint = shared.opts.sd_model_checkpoint
        self.sd_model_refiner = shared.opts.sd_model_refiner
        self.sd_model_dict = shared.opts.sd_model_dict
        self.sd_vae = shared.opts.sd_vae
        self.sd_unet = shared.opts.sd_unet
        self.sd_text_encoder = shared.opts.sd_text_encoder
        self.extra_networks_default_multiplier = shared.opts.extra_networks_default_multiplier
        self.disable_weights_auto_swap = shared.opts.disable_weights_auto_swap
        self.prompt_attention = shared.opts.prompt_attention
        shared.opts.data["disable_weights_auto_swap"] = False

    def __exit__(self, exc_type, exc_value, tb):
        #Restore overriden settings after plot generation.
        shared.opts.data["disable_weights_auto_swap"] = self.disable_weights_auto_swap
        shared.opts.data["sd_vae"] = self.vae
        shared.opts.data["schedulers_solver_order"] = self.schedulers_solver_order
        shared.opts.data["tome_ratio"] = self.tome_ratio
        shared.opts.data["todo_ratio"] = self.todo_ratio
        shared.opts.data["extra_networks_default_multiplier"] = self.extra_networks_default_multiplier
        shared.opts.data["prompt_attention"] = self.prompt_attention
        if self.sd_model_checkpoint != shared.opts.sd_model_checkpoint:
            shared.opts.data["sd_model_checkpoint"] = self.sd_model_checkpoint
            sd_models.reload_model_weights(op='model')
        if self.sd_model_refiner != shared.opts.sd_model_refiner:
            shared.opts.data["sd_model_refiner"] = self.sd_model_refiner
            sd_models.reload_model_weights(op='refiner')
        if self.sd_model_dict != shared.opts.sd_model_dict:
            shared.opts.data["sd_model_dict"] = self.sd_model_dict
            sd_models.reload_model_weights(op='dict')
        if self.sd_vae != shared.opts.sd_vae:
            shared.opts.data["sd_vae"] = self.sd_vae
            sd_vae.reload_vae_weights()
        if self.sd_text_encoder != shared.opts.sd_text_encoder:
            shared.opts.data["sd_text_encoder"] = self.sd_text_encoder
            sd_models.reload_text_encoder()
        if self.sd_unet != shared.opts.sd_unet:
            shared.opts.data["sd_unet"] = self.sd_unet
            sd_unet.load_unet(shared.sd_model)


axis_options = [
    AxisOption("Nothing", str, do_nothing, fmt=format_nothing),
    AxisOption("[Model] Model", str, apply_checkpoint, cost=1.0, fmt=format_value_add_label, choices=lambda: sorted(sd_models.checkpoints_list)),
    AxisOption("[Model] UNET", str, apply_unet, cost=0.8, choices=lambda: ['None'] + list(sd_unet.unet_dict)),
    AxisOption("[Model] VAE", str, apply_vae, cost=0.6, choices=lambda: ['None'] + list(sd_vae.vae_dict)),
    AxisOption("[Model] Refiner", str, apply_refiner, cost=0.8, fmt=format_value_add_label, choices=lambda: ['None'] + sorted(sd_models.checkpoints_list)),
    AxisOption("[Model] Text encoder", str, apply_te, cost=0.7, choices=shared_items.sd_te_items),
    AxisOption("[Model] Dictionary", str, apply_dict, fmt=format_value_add_label, cost=0.9, choices=lambda: ['None'] + list(sd_models.checkpoints_list)),
    AxisOption("[Prompt] Search & replace", str, apply_prompt, fmt=format_value_add_label),
    AxisOption("[Prompt] Prompt order", str_permutations, apply_order, fmt=format_value_join_list),
    AxisOption("[Prompt] Prompt parser", str, apply_setting("prompt_attention"), choices=lambda: ["native", "compel", "xhinker", "a1111", "fixed"]),
    AxisOption("[Network] LoRA", str, apply_lora, cost=0.5, choices=list_lora),
    AxisOption("[Network] LoRA strength", float, apply_lora_strength, cost=0.6),
    AxisOption("[Network] Styles", str, apply_styles, choices=lambda: [s.name for s in shared.prompt_styles.styles.values()]),
    AxisOption("[Param] Width", int, apply_field("width")),
    AxisOption("[Param] Height", int, apply_field("height")),
    AxisOption("[Param] Seed", int, apply_seed),
    AxisOption("[Param] Steps", int, apply_field("steps")),
    AxisOption("[Param] Guidance scale", float, apply_field("cfg_scale")),
    AxisOption("[Param] Guidance end", float, apply_field("cfg_end")),
    AxisOption("[Param] Variation seed", int, apply_field("subseed")),
    AxisOption("[Param] Variation strength", float, apply_field("subseed_strength")),
    AxisOption("[Param] Clip skip", float, apply_clip_skip),
    AxisOption("[Param] Denoising strength", float, apply_field("denoising_strength")),
    AxisOptionImg2Img("[Param] Mask weight", float, apply_field("inpainting_mask_weight")),
    AxisOption("[Process] Model args", str, apply_task_args),
    AxisOption("[Process] Processing args", str, apply_processing),
    AxisOption("[Process] Server options", str, apply_options),
    AxisOptionTxt2Img("[Sampler] Name", str, apply_sampler, fmt=format_value_add_label, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers]),
    AxisOptionImg2Img("[Sampler] Name", str, apply_sampler, fmt=format_value_add_label, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers_for_img2img]),
    AxisOption("[Sampler] Sigma method", str, apply_setting("schedulers_sigma"), choices=lambda: ['default', 'karras', 'betas', 'exponential', 'lambdas']),
    AxisOption("[Sampler] Timestep spacing", str, apply_setting("schedulers_timestep_spacing"), choices=lambda: ['default', 'linspace', 'leading', 'trailing']),
    AxisOption("[Sampler] Timestep range", int, apply_setting("schedulers_timesteps_range")),
    AxisOption("[Sampler] Solver order", int, apply_setting("schedulers_solver_order")),
    AxisOption("[Sampler] Beta schedule", str, apply_setting("schedulers_beta_schedule"), choices=lambda: ['default', 'linear', 'scaled', 'cosine']),
    AxisOption("[Sampler] Beta start", float, apply_setting("schedulers_beta_start")),
    AxisOption("[Sampler] Beta end", float, apply_setting("schedulers_beta_end")),
    AxisOption("[Sampler] Shift", float, apply_setting("schedulers_shift")),
    AxisOption("[Sampler] eta delta", float, apply_setting("eta_noise_seed_delta")),
    AxisOption("[Sampler] eta multiplier", float, apply_setting("scheduler_eta")),
    AxisOption("[Refine] Upscaler", str, apply_field("hr_upscaler"), cost=0.3, choices=lambda: [*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]),
    AxisOption("[Refine] Sampler", str, apply_hr_sampler_name, fmt=format_value_add_label, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.samplers]),
    AxisOption("[Refine] Denoising strength", float, apply_field("denoising_strength")),
    AxisOption("[Refine] Hires steps", int, apply_field("hr_second_pass_steps")),
    AxisOption("[Refine] Guidance scale", float, apply_field("image_cfg_scale")),
    AxisOption("[Refine] Guidance rescale", float, apply_field("diffusers_guidance_rescale")),
    AxisOption("[Refine] Refiner start", float, apply_field("refiner_start")),
    AxisOption("[Refine] Refiner steps", float, apply_field("refiner_steps")),
    AxisOption("[Postprocess] Upscaler", str, apply_upscaler, cost=0.4, choices=lambda: [x.name for x in shared.sd_upscalers][1:]),
    AxisOption("[Postprocess] Context", str, apply_context, choices=lambda: ["Add with forward", "Remove with forward", "Add with backward", "Remove with backward"]),
    AxisOption("[Postprocess] Detailer", str, apply_detailer, fmt=format_value_add_label),
    AxisOption("[Postprocess] Detailer strength", str, apply_field("detailer_strength")),
    AxisOption("[HDR] Mode", int, apply_field("hdr_mode")),
    AxisOption("[HDR] Brightness", float, apply_field("hdr_brightness")),
    AxisOption("[HDR] Color", float, apply_field("hdr_color")),
    AxisOption("[HDR] Sharpen", float, apply_field("hdr_sharpen")),
    AxisOption("[HDR] Clamp boundary", float, apply_field("hdr_boundary")),
    AxisOption("[HDR] Clamp threshold", float, apply_field("hdr_threshold")),
    AxisOption("[HDR] Maximize center shift", float, apply_field("hdr_max_center")),
    AxisOption("[HDR] Maximize boundary", float, apply_field("hdr_max_boundry")),
    AxisOption("[HDR] Tint color hex", str, apply_field("hdr_color_picker")),
    AxisOption("[HDR] Tint ratio", float, apply_field("hdr_tint_ratio")),
    AxisOption("[Token Merging] ToMe ratio", float, apply_setting('tome_ratio')),
    AxisOption("[Token Merging] ToDo ratio", float, apply_setting('todo_ratio')),
    AxisOption("[FreeU] 1st stage backbone factor", float, apply_setting('freeu_b1')),
    AxisOption("[FreeU] 2nd stage backbone factor", float, apply_setting('freeu_b2')),
    AxisOption("[FreeU] 1st stage skip factor", float, apply_setting('freeu_s1')),
    AxisOption("[FreeU] 2nd stage skip factor", float, apply_setting('freeu_s2')),
    AxisOption("[IP adapter] Name", str, apply_field('ip_adapter_names'), cost=1.0, choices=lambda: list(ipadapter.ADAPTERS)),
    AxisOption("[IP adapter] Scale", float, apply_field('ip_adapter_scales')),
    AxisOption("[IP adapter] Starts", float, apply_field('ip_adapter_starts')),
    AxisOption("[IP adapter] Ends", float, apply_field('ip_adapter_ends')),
    AxisOption("[HiDiffusion] T1", float, apply_override('hidiffusion_t1')),
    AxisOption("[HiDiffusion] T2", float, apply_override('hidiffusion_t2')),
    AxisOption("[HiDiffusion] Agression step", float, apply_field('hidiffusion_steps')),
    AxisOption("[PAG] Attention scale", float, apply_field('pag_scale')),
    AxisOption("[PAG] Adaptive scaling", float, apply_field('pag_adaptive')),
    AxisOption("[PAG] Applied layers", str, apply_setting('pag_apply_layers')),
]
