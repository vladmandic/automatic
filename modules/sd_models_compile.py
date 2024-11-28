import copy
import time
import logging
import torch
from modules import shared, devices, sd_models, model_quant
from installer import install, setup_logging


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
        self.req_cache = {}
        self.partitioned_modules = {}


quant_last_model_name = None
quant_last_model_device = None
deepcache_worker = None


def apply_compile_to_model(sd_model, function, options, op=None):
    if "Model" in options:
        if hasattr(sd_model, 'unet') and hasattr(sd_model.unet, 'config'):
            sd_model.unet = function(sd_model.unet, op="unet", sd_model=sd_model)
        if hasattr(sd_model, 'transformer') and hasattr(sd_model.transformer, 'config'):
            sd_model.transformer = function(sd_model.transformer, op="transformer", sd_model=sd_model)
        if hasattr(sd_model, 'decoder_pipe') and hasattr(sd_model, 'decoder'):
            sd_model.decoder = None
            sd_model.decoder = sd_model.decoder_pipe.decoder = function(sd_model.decoder_pipe.decoder, op="decoder_pipe.decoder", sd_model=sd_model)
        if hasattr(sd_model, 'prior_pipe') and hasattr(sd_model.prior_pipe, 'prior'):
            if op == "nncf" and "StableCascade" in sd_model.__class__.__name__: # fixes dtype errors
                backup_clip_txt_pooled_mapper = copy.deepcopy(sd_model.prior_pipe.prior.clip_txt_pooled_mapper)
            sd_model.prior_pipe.prior = function(sd_model.prior_pipe.prior, op="prior_pipe.prior", sd_model=sd_model)
            if op == "nncf" and "StableCascade" in sd_model.__class__.__name__:
                sd_model.prior_pipe.prior.clip_txt_pooled_mapper = backup_clip_txt_pooled_mapper
    if "Text Encoder" in options:
        if hasattr(sd_model, 'text_encoder') and hasattr(sd_model.text_encoder, 'config'):
            if hasattr(sd_model, 'decoder_pipe') and hasattr(sd_model.decoder_pipe, 'text_encoder'):
                sd_model.decoder_pipe.text_encoder = function(sd_model.decoder_pipe.text_encoder, op="decoder_pipe.text_encoder", sd_model=sd_model)
            else:
                if op == "nncf" and sd_model.text_encoder.__class__.__name__ in {"T5EncoderModel", "UMT5EncoderModel"}:
                    from modules.sd_hijack import NNCF_T5DenseGatedActDense # T5DenseGatedActDense uses fp32
                    for i in range(len(sd_model.text_encoder.encoder.block)):
                        sd_model.text_encoder.encoder.block[i].layer[1].DenseReluDense = NNCF_T5DenseGatedActDense(
                            sd_model.text_encoder.encoder.block[i].layer[1].DenseReluDense,
                            dtype=torch.float32 if devices.dtype != torch.bfloat16 else torch.bfloat16
                        )
                sd_model.text_encoder = function(sd_model.text_encoder, op="text_encoder", sd_model=sd_model)
        if hasattr(sd_model, 'text_encoder_2') and hasattr(sd_model.text_encoder_2, 'config'):
            if op == "nncf" and sd_model.text_encoder_2.__class__.__name__ in {"T5EncoderModel", "UMT5EncoderModel"}:
                from modules.sd_hijack import NNCF_T5DenseGatedActDense # T5DenseGatedActDense uses fp32
                for i in range(len(sd_model.text_encoder_2.encoder.block)):
                    sd_model.text_encoder_2.encoder.block[i].layer[1].DenseReluDense = NNCF_T5DenseGatedActDense(
                        sd_model.text_encoder_2.encoder.block[i].layer[1].DenseReluDense,
                        dtype=torch.float32 if devices.dtype != torch.bfloat16 else torch.bfloat16
                    )
            sd_model.text_encoder_2 = function(sd_model.text_encoder_2, op="text_encoder_2", sd_model=sd_model)
        if hasattr(sd_model, 'text_encoder_3') and hasattr(sd_model.text_encoder_3, 'config'):
            if op == "nncf" and sd_model.text_encoder_3.__class__.__name__ in {"T5EncoderModel", "UMT5EncoderModel"}:
                from modules.sd_hijack import NNCF_T5DenseGatedActDense # T5DenseGatedActDense uses fp32
                for i in range(len(sd_model.text_encoder_3.encoder.block)):
                    sd_model.text_encoder_3.encoder.block[i].layer[1].DenseReluDense = NNCF_T5DenseGatedActDense(
                        sd_model.text_encoder_3.encoder.block[i].layer[1].DenseReluDense,
                        dtype=torch.float32 if devices.dtype != torch.bfloat16 else torch.bfloat16
                    )
            sd_model.text_encoder_3 = function(sd_model.text_encoder_3, op="text_encoder_3", sd_model=sd_model)
        if hasattr(sd_model, 'prior_pipe') and hasattr(sd_model.prior_pipe, 'text_encoder'):
            sd_model.prior_pipe.text_encoder = function(sd_model.prior_pipe.text_encoder, op="prior_pipe.text_encoder", sd_model=sd_model)
    if "VAE" in options:
        if hasattr(sd_model, 'vae') and hasattr(sd_model.vae, 'decode'):
            if op == "compile":
                sd_model.vae.decode = function(sd_model.vae.decode, op="vae_decode", sd_model=sd_model)
                sd_model.vae.encode = function(sd_model.vae.encode, op="vae_encode", sd_model=sd_model)
            else:
                sd_model.vae = function(sd_model.vae, op="vae", sd_model=sd_model)
        if hasattr(sd_model, 'movq') and hasattr(sd_model.movq, 'decode'):
            if op == "compile":
                sd_model.movq.decode = function(sd_model.movq.decode, op="movq_decode", sd_model=sd_model)
                sd_model.movq.encode = function(sd_model.movq.encode, op="movq_encode", sd_model=sd_model)
            else:
                sd_model.movq = function(sd_model.movq, op="movq", sd_model=sd_model)
        if hasattr(sd_model, 'vqgan') and hasattr(sd_model.vqgan, 'decode'):
            if op == "compile":
                sd_model.vqgan.decode = function(sd_model.vqgan.decode, op="vqgan_decode", sd_model=sd_model)
                sd_model.vqgan.encode = function(sd_model.vqgan.encode, op="vqgan_encode", sd_model=sd_model)
            else:
                sd_model.vqgan = function(sd_model.vqgan, op="vqgan", sd_model=sd_model)
            if hasattr(sd_model, 'decoder_pipe') and hasattr(sd_model.decoder_pipe, 'vqgan'):
                if op == "compile":
                    sd_model.decoder_pipe.vqgan.decode = function(sd_model.decoder_pipe.vqgan.decode, op="vqgan_decode", sd_model=sd_model)
                    sd_model.decoder_pipe.vqgan.encode = function(sd_model.decoder_pipe.vqgan.encode, op="vqgan_encode", sd_model=sd_model)
                else:
                    sd_model.decoder_pipe.vqgan = sd_model.vqgan
        if hasattr(sd_model, 'image_encoder') and hasattr(sd_model.image_encoder, 'config'):
            sd_model.image_encoder = function(sd_model.image_encoder, op="image_encoder", sd_model=sd_model)

    return sd_model


def ipex_optimize(sd_model):
    try:
        t0 = time.time()

        def ipex_optimize_model(model, op=None, sd_model=None): # pylint: disable=unused-argument
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

        sd_model = apply_compile_to_model(sd_model, ipex_optimize_model, shared.opts.ipex_optimize, op="ipex")

        t1 = time.time()
        shared.log.info(f"IPEX Optimize: time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"IPEX Optimize: error: {e}")
    return sd_model


def nncf_send_to_device(model):
    for child in model.children():
        if child.__class__.__name__ == "WeightsDecompressor":
            child.scale = child.scale.to(devices.device)
            child.zero_point = child.zero_point.to(devices.device)
        nncf_send_to_device(child)


def nncf_compress_model(model, op=None, sd_model=None):
    import nncf
    global quant_last_model_name, quant_last_model_device # pylint: disable=global-statement
    model.eval()
    backup_embeddings = None
    if hasattr(model, "get_input_embeddings"):
        backup_embeddings = copy.deepcopy(model.get_input_embeddings())
    model = nncf.compress_weights(model)
    nncf_send_to_device(model)
    if hasattr(model, "set_input_embeddings") and backup_embeddings is not None:
        model.set_input_embeddings(backup_embeddings)
    if op is not None and shared.opts.quant_shuffle_weights:
        if quant_last_model_name is not None:
            if "." in quant_last_model_name:
                last_model_names = quant_last_model_name.split(".")
                getattr(getattr(sd_model, last_model_names[0]), last_model_names[1]).to(quant_last_model_device)
            else:
                getattr(sd_model, quant_last_model_name).to(quant_last_model_device)
            devices.torch_gc(force=True)
        if shared.cmd_opts.medvram or shared.cmd_opts.lowvram or shared.opts.diffusers_offload_mode != "none":
            quant_last_model_name = op
            quant_last_model_device = model.device
        else:
            quant_last_model_name = None
            quant_last_model_device = None
        model.to(devices.device)
    devices.torch_gc(force=True)
    return model


def nncf_compress_weights(sd_model):
    try:
        t0 = time.time()
        shared.log.info(f"Quantization: type=NNCF modules={shared.opts.nncf_compress_weights}")
        global quant_last_model_name, quant_last_model_device # pylint: disable=global-statement
        install('nncf==2.7.0', quiet=True)

        sd_model = apply_compile_to_model(sd_model, nncf_compress_model, shared.opts.nncf_compress_weights, op="nncf")
        if quant_last_model_name is not None:
            if "." in quant_last_model_name:
                last_model_names = quant_last_model_name.split(".")
                getattr(getattr(sd_model, last_model_names[0]), last_model_names[1]).to(quant_last_model_device)
            else:
                getattr(sd_model, quant_last_model_name).to(quant_last_model_device)
            devices.torch_gc(force=True)
        quant_last_model_name = None
        quant_last_model_device = None

        t1 = time.time()
        shared.log.info(f"Quantization: type=NNCF time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"Quantization: type=NNCF {e}")
    return sd_model


def optimum_quanto_model(model, op=None, sd_model=None, weights=None, activations=None):
    quanto = model_quant.load_quanto('Compile model: type=Optimum Quanto')
    global quant_last_model_name, quant_last_model_device # pylint: disable=global-statement
    if sd_model is not None and "Flux" in sd_model.__class__.__name__: # LayerNorm is not supported
        exclude_list = ["transformer_blocks.*.norm1.norm", "transformer_blocks.*.norm2", "transformer_blocks.*.norm1_context.norm", "transformer_blocks.*.norm2_context", "single_transformer_blocks.*.norm.norm", "norm_out.norm"]
    else:
        exclude_list = None
    weights = getattr(quanto, weights) if weights is not None else getattr(quanto, shared.opts.optimum_quanto_weights_type)
    if activations is not None:
        activations = getattr(quanto, activations) if activations != 'none' else None
    elif shared.opts.optimum_quanto_activations_type != 'none':
        activations = getattr(quanto, shared.opts.optimum_quanto_activations_type)
    else:
        activations = None
    model.eval()
    backup_embeddings = None
    if hasattr(model, "get_input_embeddings"):
        backup_embeddings = copy.deepcopy(model.get_input_embeddings())
    quanto.quantize(model, weights=weights, activations=activations, exclude=exclude_list)
    quanto.freeze(model)
    if hasattr(model, "set_input_embeddings") and backup_embeddings is not None:
        model.set_input_embeddings(backup_embeddings)
    if op is not None and shared.opts.quant_shuffle_weights:
        if quant_last_model_name is not None:
            if "." in quant_last_model_name:
                last_model_names = quant_last_model_name.split(".")
                getattr(getattr(sd_model, last_model_names[0]), last_model_names[1]).to(quant_last_model_device)
            else:
                getattr(sd_model, quant_last_model_name).to(quant_last_model_device)
            devices.torch_gc(force=True)
        if shared.cmd_opts.medvram or shared.cmd_opts.lowvram or shared.opts.diffusers_offload_mode != "none":
            quant_last_model_name = op
            quant_last_model_device = model.device
        else:
            quant_last_model_name = None
            quant_last_model_device = None
        model.to(devices.device)
    devices.torch_gc(force=True)
    return model


def optimum_quanto_weights(sd_model):
    try:
        if shared.opts.diffusers_offload_mode in {"balanced", "sequential"}:
            shared.log.warning(f"Quantization: type=Optimum.quanto offload={shared.opts.diffusers_offload_mode} not compatible")
            return sd_model
        t0 = time.time()
        shared.log.info(f"Quantization: type=Optimum.quanto: modules={shared.opts.optimum_quanto_weights}")
        global quant_last_model_name, quant_last_model_device # pylint: disable=global-statement
        quanto = model_quant.load_quanto()
        quanto.tensor.qbits.QBitsTensor.create = lambda *args, **kwargs: quanto.tensor.qbits.QBitsTensor(*args, **kwargs)

        sd_model = apply_compile_to_model(sd_model, optimum_quanto_model, shared.opts.optimum_quanto_weights, op="optimum-quanto")
        if quant_last_model_name is not None:
            if "." in quant_last_model_name:
                last_model_names = quant_last_model_name.split(".")
                getattr(getattr(sd_model, last_model_names[0]), last_model_names[1]).to(quant_last_model_device)
            else:
                getattr(sd_model, quant_last_model_name).to(quant_last_model_device)
            devices.torch_gc(force=True)
        quant_last_model_name = None
        quant_last_model_device = None

        if shared.opts.optimum_quanto_activations_type != 'none':
            activations = getattr(quanto, shared.opts.optimum_quanto_activations_type)
        else:
            activations = None

        if activations is not None:
            def optimum_quanto_freeze(model, op=None, sd_model=None): # pylint: disable=unused-argument
                quanto.freeze(model)
                return model
            if shared.opts.diffusers_offload_mode == "model":
                sd_model.enable_model_cpu_offload(device=devices.device)
                if hasattr(sd_model, "encode_prompt"):
                    original_encode_prompt = sd_model.encode_prompt
                    def encode_prompt(*args, **kwargs):
                        embeds = original_encode_prompt(*args, **kwargs)
                        sd_model.maybe_free_model_hooks() # Diffusers keeps the TE on VRAM
                        return embeds
                    sd_model.encode_prompt = encode_prompt
            else:
                sd_models.move_model(sd_model, devices.device)
            with quanto.Calibration(momentum=0.9):
                sd_model(prompt="dummy prompt", num_inference_steps=10)
            sd_model = apply_compile_to_model(sd_model, optimum_quanto_freeze, shared.opts.optimum_quanto_weights, op="optimum-quanto-freeze")
            if shared.opts.diffusers_offload_mode == "model":
                sd_models.disable_offload(sd_model)
                sd_models.move_model(sd_model, devices.cpu)
                if hasattr(sd_model, "encode_prompt"):
                    sd_model.encode_prompt = original_encode_prompt
            devices.torch_gc(force=True)

        t1 = time.time()
        shared.log.info(f"Quantization: type=Optimum.quanto time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"Quantization: type=Optimum.quanto {e}")
    return sd_model


def optimize_openvino(sd_model):
    try:
        from modules.intel.openvino import openvino_fx # pylint: disable=unused-import
        torch._dynamo.eval_frame.check_if_dynamo_supported = lambda: True # pylint: disable=protected-access
        if shared.compiled_model_state is not None:
            shared.compiled_model_state.compiled_cache.clear()
            shared.compiled_model_state.req_cache.clear()
            shared.compiled_model_state.partitioned_modules.clear()
        shared.compiled_model_state = CompiledModelState()
        shared.compiled_model_state.is_compiled = True
        shared.compiled_model_state.first_pass = True if not shared.opts.cuda_compile_precompile else False
        shared.compiled_model_state.first_pass_vae = True if not shared.opts.cuda_compile_precompile else False
        shared.compiled_model_state.first_pass_refiner = True if not shared.opts.cuda_compile_precompile else False
        sd_models.set_accelerate(sd_model)
    except Exception as e:
        shared.log.warning(f"Model compile: task=OpenVINO: {e}")
    return sd_model


def compile_onediff(sd_model):
    try:
        from onediff.infer_compiler import oneflow_compile

    except Exception as e:
        shared.log.warning(f"Model compile: task=onediff {e}")
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
        shared.log.info(f"Model compile: task=onediff time={t1-t0:.2f}")
    except Exception as e:
        shared.log.info(f"Model compile: task=onediff {e}")
    return sd_model


def compile_stablefast(sd_model):
    try:
        import sfast.compilers.stable_diffusion_pipeline_compiler as sf
    except Exception as e:
        shared.log.warning(f'Model compile: task=stablefast: {e}')
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
        shared.log.info(f"Model compile: task=stablefast config={config.__dict__} time={t1-t0:.2f}")
    except Exception as e:
        shared.log.info(f"Model compile: task=stablefast {e}")
    return sd_model


def compile_torch(sd_model):
    try:
        t0 = time.time()
        import torch._dynamo # pylint: disable=unused-import,redefined-outer-name
        torch._dynamo.reset() # pylint: disable=protected-access
        shared.log.debug(f"Model compile: task=torch backends={torch._dynamo.list_backends()}") # pylint: disable=protected-access

        def torch_compile_model(model, op=None, sd_model=None): # pylint: disable=unused-argument
            if hasattr(model, "device") and model.device.type != "meta":
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
            shared.log.error(f"Model compile: torch inductor config error: {e}")

        sd_model = apply_compile_to_model(sd_model, function=torch_compile_model, options=shared.opts.cuda_compile, op="compile")

        setup_logging() # compile messes with logging so reset is needed
        if shared.opts.cuda_compile_precompile:
            sd_model("dummy prompt")
        t1 = time.time()
        shared.log.info(f"Model compile: task=torch time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"Model compile: task=torch {e}")
    return sd_model


def check_deepcache(enable: bool):
    if deepcache_worker is not None:
        if enable:
            deepcache_worker.enable()
        else:
            deepcache_worker.disable()


def compile_deepcache(sd_model):
    global deepcache_worker # pylint: disable=global-statement
    if not hasattr(sd_model, 'unet'):
        shared.log.warning(f'Model compile: task=deepcache pipeline={sd_model.__class__} not supported')
        return sd_model
    try:
        from DeepCache import DeepCacheSDHelper
    except Exception as e:
        shared.log.warning(f'Model compile: task=deepcache {e}')
        return sd_model
    t0 = time.time()
    check_deepcache(False)
    deepcache_worker = DeepCacheSDHelper(pipe=sd_model)
    deepcache_worker.set_params(cache_interval=shared.opts.deep_cache_interval, cache_branch_id=0)
    t1 = time.time()
    shared.log.info(f"Model compile: task=deepcache config={deepcache_worker.params} time={t1-t0:.2f}")
    # config={'cache_interval': 3, 'cache_layer_id': 0, 'cache_block_id': 0, 'skip_mode': 'uniform'} time=0.00
    return sd_model


def compile_diffusers(sd_model):
    if 'Model' not in shared.opts.cuda_compile:
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


def torchao_quantization(sd_model):
    try:
        install('torchao', quiet=True)
        from torchao import quantization as q
    except Exception as e:
        shared.log.error(f"Quantization: type=TorchAO quantization not supported: {e}")
        return sd_model
    if shared.opts.torchao_quantization_type == "int8+act":
        fn = q.int8_dynamic_activation_int8_weight
    elif shared.opts.torchao_quantization_type == "int8":
        fn = q.int8_weight_only
    elif shared.opts.torchao_quantization_type == "int4":
        fn = q.int4_weight_only
    elif shared.opts.torchao_quantization_type == "fp8+act":
        fn = q.float8_dynamic_activation_float8_weight
    elif shared.opts.torchao_quantization_type == "fp8":
        fn = q.float8_weight_only
    elif shared.opts.torchao_quantization_type == "fpx":
        fn = q.fpx_weight_only
    else:
        shared.log.error(f"Quantization: type=TorchAO type={shared.opts.torchao_quantization_type} not supported")
        return sd_model
    shared.log.info(f"Quantization: type=TorchAO pipe={sd_model.__class__.__name__} quant={shared.opts.torchao_quantization_type} fn={fn} targets={shared.opts.torchao_quantization}")
    try:
        t0 = time.time()
        modules = []
        if hasattr(sd_model, 'unet') and 'Model' in shared.opts.torchao_quantization:
            modules.append('unet')
            q.quantize_(sd_model.unet, fn(), device=devices.device)
        if hasattr(sd_model, 'transformer') and 'Model' in shared.opts.torchao_quantization:
            modules.append('transformer')
            q.quantize_(sd_model.transformer, fn(), device=devices.device)
        if hasattr(sd_model, 'vae') and 'VAE' in shared.opts.torchao_quantization:
            modules.append('vae')
            q.quantize_(sd_model.vae, fn(), device=devices.device)
        if hasattr(sd_model, 'text_encoder') and 'Text Encoder' in shared.opts.torchao_quantization:
            modules.append('te1')
            q.quantize_(sd_model.text_encoder, fn(), device=devices.device)
        if hasattr(sd_model, 'text_encoder_2') and 'Text Encoder' in shared.opts.torchao_quantization:
            modules.append('te2')
            q.quantize_(sd_model.text_encoder_2, fn(), device=devices.device)
        if hasattr(sd_model, 'text_encoder_3') and 'Text Encoder' in shared.opts.torchao_quantization:
            modules.append('te3')
            q.quantize_(sd_model.text_encoder_3, fn(), device=devices.device)
        t1 = time.time()
        shared.log.info(f"Quantization: type=TorchAO modules={modules} time={t1-t0:.2f}")
    except Exception as e:
        shared.log.error(f"Quantization: type=TorchAO {e}")
    setup_logging() # torchao uses dynamo which messes with logging so reset is needed
    return sd_model


def openvino_recompile_model(p, hires=False, refiner=False): # recompile if a parameter changes
    if 'Model' in shared.opts.cuda_compile and shared.opts.cuda_compile_backend != 'none':
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
    if 'Model' in shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx":
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
