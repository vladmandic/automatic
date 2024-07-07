import torch
import transformers
import diffusers


repo_id = 'Kwai-Kolors/Kolors'
encoder_id = 'THUDM/chatglm3-6b'


def load_kolors(_checkpoint_info, diffusers_load_config={}):
    from modules import shared, devices, modelloader
    modelloader.hf_login()
    diffusers_load_config['variant'] = "fp16"
    if 'torch_dtype' not in diffusers_load_config:
        diffusers_load_config['torch_dtype'] = 'torch.float16'

    text_encoder = transformers.AutoModel.from_pretrained(encoder_id, torch_dtype=torch.float16, trust_remote_code=True, cache_dir=shared.opts.diffusers_dir)
    # text_encoder = transformers.AutoModel.from_pretrained("THUDM/chatglm3-6b", torch_dtype=torch.float16, trust_remote_code=True).quantize(4).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_id, trust_remote_code=True, cache_dir=shared.opts.diffusers_dir)
    pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
        repo_id,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        cache_dir = shared.opts.diffusers_dir,
        **diffusers_load_config,
    )
    devices.torch_gc()
    return pipe
