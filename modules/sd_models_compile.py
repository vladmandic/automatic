import copy
import time
import logging
import torch
from modules import shared, devices, sd_models
from installer import setup_logging


#Used by OpenVINO, can be used with TensorRT or Olive
class CompiledModelState:
    def __init__(self):
        self.is_compiled = False
        self.model_hash_str = ""
        self.first_pass = True
        self.first_pass_refiner = True
        self.first_pass_vae = True
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.partition_id = 0
        self.cn_model = []
        self.lora_model = []
        self.compiled_cache = {}
        self.partitioned_modules = {}


deepcache_worker = None


def apply_compile_to_model(sd_model, function, options):
    if "Model" in options:
        if hasattr(sd_model, 'unet') and hasattr(sd_model.unet, 'config'):
            sd_model.unet = function(sd_model.unet)
        if hasattr(sd_model, 'transformer') and hasattr(sd_model.transformer, 'config'):
            sd_model.transformer = function(sd_model.transformer)
        if hasattr(sd_model, 'decoder_pipe') and hasattr(sd_model, 'decoder'):
            sd_model.decoder = None
            sd_model.decoder = sd_model.decoder_pipe.decoder = function(sd_model.decoder_pipe.decoder)
        if hasattr(sd_model, 'prior_pipe') and hasattr(sd_model, 'prior_prior'):
            sd_model.prior_prior = None
            sd_model.prior_prior = sd_model.prior_pipe.prior = function(sd_model.prior_pipe.prior)
    if "VAE" in options:
        if hasattr(sd_model, 'vae') and hasattr(sd_model.vae, 'decode'):
            sd_model.vae = function(sd_model.vae)
        if hasattr(sd_model, 'movq') and hasattr(sd_model.movq, 'decode'):
            sd_model.movq = function(sd_model.movq)
        if hasattr(sd_model, 'vqgan') and hasattr(sd_model.vqgan, 'decode'):
            sd_model.vqgan = function(sd_model.vqgan)
        if hasattr(sd_model, 'image_encoder') and hasattr(sd_model.image_encoder, 'config'):
            sd_model.image_encoder = function(sd_model.image_encoder)
    if "Text Encoder" in options:
        if hasattr(sd_model, 'text_encoder') and hasattr(sd_model.text_encoder, 'config'):
            if hasattr(sd_model, 'decoder_pipe'):
                sd_model.text_encoder = None
                sd_model.text_encoder = sd_model.decoder_pipe.text_encoder = function(sd_model.decoder_pipe.text_encoder)
            else:
                sd_model.text_encoder = function(sd_model.text_encoder)
        if hasattr(sd_model, 'text_encoder_2') and hasattr(sd_model.text_encoder_2, 'config'):
            sd_model.text_encoder_2 = function(sd_model.text_encoder_2)
        if hasattr(sd_model, 'prior_pipe') and hasattr(sd_model, 'prior_text_encoder'):
            sd_model.prior_text_encoder = None
            sd_model.prior_text_encoder = sd_model.prior_pipe.text_encoder = function(sd_model.prior_pipe.text_encoder)

    return sd_model


def ipex_optimize(sd_model):
    try:
        t0 = time.time()

        def ipex_optimize_model(model):
            import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
            model.eval()
            model.training = False
            if model.device.type != "meta":
                return_device = model.device
                model = ipex.optimize(model.to(devices.device),
                    dtype=devices.dtype,
                    inplace=True,
                    weights_prepack=False
                ).to(return_device) # pylint: disable=attribute-defined-outside-init
            else:
                model = ipex.optimize(model,
                    dtype=devices.dtype,
                    inplace=True,
                    weights_prepack=False
                ) # pylint: disable=attribute-defined-outside-init
            devices.torch_gc()
            return model

        sd_model = apply_compile_to_model(sd_model, ipex_optimize_model, shared.opts.ipex_optimize)

        t1 = time.time()
        shared.log.info(f"IPEX Optimize: time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"IPEX Optimize: error: {e}")
    return sd_model


def nncf_compress_weights(sd_model):
    try:
        t0 = time.time()
        if sd_model.device.type == "meta":
            shared.log.warning("Compress Weights is not compatible with Sequential CPU offload")
            return sd_model

        def nncf_compress_model(model):
            return_device = model.device
            model.eval()
            backup_embeddings = None
            if hasattr(model, "get_input_embeddings"):
                backup_embeddings = copy.deepcopy(model.get_input_embeddings())
            model = nncf.compress_weights(model.to(devices.device)).to(return_device)
            if hasattr(model, "set_input_embeddings") and backup_embeddings is not None:
                model.set_input_embeddings(backup_embeddings)
            devices.torch_gc(force=True)
            return model

        import nncf
        shared.compiled_model_state = CompiledModelState()
        shared.compiled_model_state.is_compiled = True

        sd_model = apply_compile_to_model(sd_model, nncf_compress_model, shared.opts.nncf_compress_weights)

        t1 = time.time()
        shared.log.info(f"Compress Weights: time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"Compress Weights: error: {e}")
    return sd_model


def optimize_openvino(sd_model):
    try:
        from modules.intel.openvino import openvino_fx # pylint: disable=unused-import
        torch._dynamo.eval_frame.check_if_dynamo_supported = lambda: True # pylint: disable=protected-access
        if shared.compiled_model_state is not None:
            shared.compiled_model_state.compiled_cache.clear()
            shared.compiled_model_state.partitioned_modules.clear()
        shared.compiled_model_state = CompiledModelState()
        shared.compiled_model_state.is_compiled = True
        shared.compiled_model_state.first_pass = True if not shared.opts.cuda_compile_precompile else False
        shared.compiled_model_state.first_pass_vae = True if not shared.opts.cuda_compile_precompile else False
        shared.compiled_model_state.first_pass_refiner = True if not shared.opts.cuda_compile_precompile else False
        sd_model.has_accelerate = True
    except Exception as e:
        shared.log.warning(f"Model compile: task=OpenVINO: {e}")
    return sd_model


def compile_onediff(sd_model):
    try:
        from onediff.infer_compiler import oneflow_compile

    except Exception as e:
        shared.log.warning(f"Model compile using onediff/oneflow: {e}")
        return sd_model

    try:
        t0 = time.time()
        # For some reason compiling the text_encoder, when it is used by
        # the 'compel' package which sdnext uses, it becomes 100 times
        # slower as if it is recompiling every time.
        #sd_model.text_encoder = oneflow_compile(sd_model.text_encoder)
        #if hasattr(sd_model, 'text_endcoder_2'):
        #    sd_model.text_encoder_2 = oneflow_compile(sd_model.text_encoder_2)
        sd_model.unet = oneflow_compile(sd_model.unet)
        sd_model.vae.encoder = oneflow_compile(sd_model.vae.encoder)
        sd_model.vae.decoder = oneflow_compile(sd_model.vae.decoder)
        # How are Loras, Adaptors, and other things compiled

        # DW: I'm unclear whether this is also a problem with onediff
        # as it was for sfast.
        setup_logging() # compile messes with logging so reset is needed
        if shared.opts.cuda_compile_precompile:
            sd_model("dummy prompt")
        t1 = time.time()
        shared.log.info(f"Model compile: task=onediff/oneflow time={t1-t0:.2f}")
    except Exception as e:
        shared.log.info(f"Model compile: task=onediff/oneflow error: {e}")
    return sd_model


def compile_stablefast(sd_model):
    try:
        import sfast.compilers.stable_diffusion_pipeline_compiler as sf
    except Exception as e:
        shared.log.warning(f'Model compile using stable-fast: {e}')
        return sd_model
    config = sf.CompilationConfig.Default()
    try:
        import xformers # pylint: disable=unused-import
        config.enable_xformers = True
    except Exception:
        pass
    try:
        import triton # pylint: disable=unused-import
        config.enable_triton = True
    except Exception:
        pass
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    config.enable_cuda_graph = shared.opts.cuda_compile_fullgraph
    config.enable_jit_freeze = shared.opts.diffusers_eval
    config.memory_format = torch.channels_last if shared.opts.opt_channelslast else torch.contiguous_format
    # config.trace_scheduler = False
    # config.enable_cnn_optimization
    # config.prefer_lowp_gemm
    try:
        t0 = time.time()
        sd_model = sf.compile(sd_model, config)
        sd_model.sfast = True
        setup_logging() # compile messes with logging so reset is needed
        if shared.opts.cuda_compile_precompile:
            sd_model("dummy prompt")
        t1 = time.time()
        shared.log.info(f"Model compile: task='Stable-fast' config={config.__dict__} time={t1-t0:.2f}")
    except Exception as e:
        shared.log.info(f"Model compile: task=Stable-fast error: {e}")
    return sd_model


def compile_torch(sd_model):
    try:
        t0 = time.time()
        import torch._dynamo # pylint: disable=unused-import,redefined-outer-name
        torch._dynamo.reset() # pylint: disable=protected-access
        shared.log.debug(f"Model compile available backends: {torch._dynamo.list_backends()}") # pylint: disable=protected-access

        def torch_compile_model(model):
            if model.device.type != "meta":
                return_device = model.device
                model = torch.compile(model.to(devices.device),
                    mode=shared.opts.cuda_compile_mode,
                    backend=shared.opts.cuda_compile_backend,
                    fullgraph=shared.opts.cuda_compile_fullgraph
                ).to(return_device)
            else:
                model = torch.compile(model,
                    mode=shared.opts.cuda_compile_mode,
                    backend=shared.opts.cuda_compile_backend,
                    fullgraph=shared.opts.cuda_compile_fullgraph
                )
            devices.torch_gc()
            return model

        if shared.opts.cuda_compile_backend == "openvino_fx":
            sd_model = optimize_openvino(sd_model)
        elif shared.opts.cuda_compile_backend == "olive-ai":
            if shared.compiled_model_state is None:
                shared.compiled_model_state = CompiledModelState()
            return sd_model
        elif shared.opts.cuda_compile_backend ==  "migraphx":
            import torch_migraphx # pylint: disable=unused-import
        log_level = logging.WARNING if shared.opts.cuda_compile_verbose else logging.CRITICAL # pylint: disable=protected-access
        if hasattr(torch, '_logging'):
            torch._logging.set_logs(dynamo=log_level, aot=log_level, inductor=log_level) # pylint: disable=protected-access
        torch._dynamo.config.verbose = shared.opts.cuda_compile_verbose # pylint: disable=protected-access
        torch._dynamo.config.suppress_errors = shared.opts.cuda_compile_errors # pylint: disable=protected-access

        try:
            torch._inductor.config.conv_1x1_as_mm = True # pylint: disable=protected-access
            torch._inductor.config.coordinate_descent_tuning = True # pylint: disable=protected-access
            torch._inductor.config.epilogue_fusion = False # pylint: disable=protected-access
            torch._inductor.config.coordinate_descent_check_all_directions = True # pylint: disable=protected-access
            torch._inductor.config.use_mixed_mm = True # pylint: disable=protected-access
            # torch._inductor.config.force_fuse_int_mm_with_mul = True # pylint: disable=protected-access
        except Exception as e:
            shared.log.error(f"Torch inductor config error: {e}")

        sd_model = apply_compile_to_model(sd_model, torch_compile_model, shared.opts.cuda_compile)

        setup_logging() # compile messes with logging so reset is needed
        if shared.opts.cuda_compile_precompile:
            sd_model("dummy prompt")
        t1 = time.time()
        shared.log.info(f"Model compile: time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"Model compile error: {e}")
    return sd_model


def check_deepcache(enable: bool):
    if deepcache_worker is not None:
        if enable:
            deepcache_worker.enable()
        else:
            deepcache_worker.disable()


def compile_deepcache(sd_model):
    global deepcache_worker # pylint: disable=global-statement
    try:
        from DeepCache import DeepCacheSDHelper
    except Exception as e:
        shared.log.warning(f'Model compile using deep-cache: {e}')
        return sd_model
    t0 = time.time()
    check_deepcache(False)
    deepcache_worker = DeepCacheSDHelper(pipe=sd_model)
    deepcache_worker.set_params(cache_interval=shared.opts.deep_cache_interval, cache_branch_id=0)
    t1 = time.time()
    shared.log.info(f"Model compile: task='DeepCache' config={deepcache_worker.params} time={t1-t0:.2f}")
    # config={'cache_interval': 3, 'cache_layer_id': 0, 'cache_block_id': 0, 'skip_mode': 'uniform'} time=0.00
    return sd_model


def compile_diffusers(sd_model):
    if not shared.opts.cuda_compile:
        return sd_model
    if shared.opts.cuda_compile_backend == 'none':
        shared.log.warning('Model compile enabled but no backend specified')
        return sd_model
    shared.log.info(f"Model compile: pipeline={sd_model.__class__.__name__} mode={shared.opts.cuda_compile_mode} backend={shared.opts.cuda_compile_backend} fullgraph={shared.opts.cuda_compile_fullgraph} compile={shared.opts.cuda_compile}")
    if shared.opts.cuda_compile_backend == 'onediff':
        sd_model = compile_onediff(sd_model)
    elif shared.opts.cuda_compile_backend == 'stable-fast':
        sd_model = compile_stablefast(sd_model)
    elif shared.opts.cuda_compile_backend == 'deep-cache':
        sd_model = compile_deepcache(sd_model)
    else:
        check_deepcache(False)
        sd_model = compile_torch(sd_model)
    return sd_model


def dynamic_quantization(sd_model):
    try:
        from torchao.quantization import quant_api
    except Exception as e:
        shared.log.error(f"Model dynamic quantization not supported: {e}")
        return sd_model

    def dynamic_quant_filter_fn(mod, *args): # pylint: disable=unused-argument
        return (isinstance(mod, torch.nn.Linear) and mod.in_features > 16 and (mod.in_features, mod.out_features)
                not in [(1280, 640), (1920, 1280), (1920, 640), (2048, 1280), (2048, 2560), (2560, 1280), (256, 128), (2816, 1280), (320, 640), (512, 1536), (512, 256), (512, 512), (640, 1280), (640, 1920), (640, 320), (640, 5120), (640, 640), (960, 320), (960, 640)])

    def conv_filter_fn(mod, *args): # pylint: disable=unused-argument
        return (isinstance(mod, torch.nn.Conv2d) and mod.kernel_size == (1, 1) and 128 in [mod.in_channels, mod.out_channels])

    shared.log.info(f"Model dynamic quantization: pipeline={sd_model.__class__.__name__}")
    try:
        quant_api.swap_conv2d_1x1_to_linear(sd_model.unet, conv_filter_fn)
        quant_api.swap_conv2d_1x1_to_linear(sd_model.vae, conv_filter_fn)
        quant_api.apply_dynamic_quant(sd_model.unet, dynamic_quant_filter_fn)
        quant_api.apply_dynamic_quant(sd_model.vae, dynamic_quant_filter_fn)
    except Exception as e:
        shared.log.error(f"Model dynamic quantization error: {e}")
    return sd_model


def openvino_recompile_model(p, hires=False, refiner=False): # recompile if a parameter changes
    if shared.opts.cuda_compile and shared.opts.cuda_compile_backend != 'none':
        if shared.opts.cuda_compile_backend == "openvino_fx":
            compile_height = p.height if not hires and hasattr(p, 'height') else p.hr_upscale_to_y
            compile_width = p.width if not hires and hasattr(p, 'width') else p.hr_upscale_to_x
            if (shared.compiled_model_state is None or
            (not shared.compiled_model_state.first_pass
            and (shared.compiled_model_state.height != compile_height
            or shared.compiled_model_state.width != compile_width
            or shared.compiled_model_state.batch_size != p.batch_size))):
                if refiner:
                    shared.log.info("OpenVINO: Recompiling refiner")
                    sd_models.unload_model_weights(op='refiner')
                    sd_models.reload_model_weights(op='refiner')
                else:
                    shared.log.info("OpenVINO: Recompiling base model")
                    sd_models.unload_model_weights(op='model')
                    sd_models.reload_model_weights(op='model')
            shared.compiled_model_state.height = compile_height
            shared.compiled_model_state.width = compile_width
            shared.compiled_model_state.batch_size = p.batch_size


def openvino_post_compile(op="base"): # delete unet after OpenVINO compile
    if shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx":
        if shared.compiled_model_state.first_pass and op == "base":
            shared.compiled_model_state.first_pass = False
            if not shared.opts.openvino_disable_memory_cleanup and hasattr(shared.sd_model, "unet"):
                shared.sd_model.unet.apply(sd_models.convert_to_faketensors)
                devices.torch_gc(force=True)
        if shared.compiled_model_state.first_pass_refiner and op == "refiner":
            shared.compiled_model_state.first_pass_refiner = False
            if not shared.opts.openvino_disable_memory_cleanup and hasattr(shared.sd_refiner, "unet"):
                shared.sd_refiner.unet.apply(sd_models.convert_to_faketensors)
                devices.torch_gc(force=True)
