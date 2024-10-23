import os
from typing import List, Union
from PIL import Image
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from diffusers.models import AutoencoderKL
from diffusers.utils import replace_example_docstring
from .model import OmniGen
from .processor import OmniGenProcessor
from .scheduler import OmniGenScheduler


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from OmniGen import OmniGenPipeline
        >>> pipe = FluxControlNetPipeline.from_pretrained(
        ...     base_model
        ... )
        >>> prompt = "A woman holds a bouquet of flowers and faces the camera"
        >>> image = pipe(
        ...     prompt,
        ...     guidance_scale=3.0,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("t2i.png")
        ```
"""


class OmniGenPipeline():
    def __init__(
        self,
        vae: AutoencoderKL,
        model: OmniGen,
        processor: OmniGenProcessor,
    ):
        super().__init__()
        self.vae = vae
        self.model = model
        self.processor = processor
        self.device = None
        self.dtype: None
        self.separate_cfg_infer: bool = True
        self.use_kv_cache: bool = False
        # omnigen does not inherit from diffusionpipeline so we hack it
        self._internal_dict = { # pylint: disable=protected-access
            'vae': self.vae,
            'model': self.model,
            'processor': self.processor,
        }

    @classmethod
    def from_pretrained(cls, model_name, vae_path: str=None, cache_dir: str=None):
        if not os.path.exists(model_name):
            cache_dir = cache_dir or os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                           cache_dir=cache_dir,
                                           ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])
        model = OmniGen.from_pretrained(model_name)
        processor = OmniGenProcessor.from_pretrained(model_name)
        if os.path.exists(os.path.join(model_name, "vae")):
            vae = AutoencoderKL.from_pretrained(os.path.join(model_name, "vae"))
        else:
            vae = AutoencoderKL.from_pretrained(vae_path or "stabilityai/sdxl-vae")
        return cls(vae, model, processor)

    def merge_lora(self, lora_path: str):
        model = PeftModel.from_pretrained(self.model, lora_path)
        model.merge_and_unload()
        self.model = model

    def to(self, device: Union[str, torch.device]):
        if isinstance(device, str):
            device = torch.device(device)
        self.model.to(device)
        self.vae.to(device)

    def vae_encode(self, x, dtype):
        x = x.to(dtype)
        if self.vae.config.shift_factor is not None:
            x = self.vae.encode(x).latent_dist.sample()
            x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            x = self.vae.encode(x).latent_dist.sample().mul_(self.vae.config.scaling_factor)
        x = x.to(dtype)
        return x

    def move_to_device(self, data):
        if isinstance(data, list):
            return [x.to(self.device) for x in data]
        return data.to(self.device)


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        input_images: Union[List[str], List[List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        output_type: str = 'latent',
        seed: int = None,
        ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation. 
            input_images (`List[str]` or `List[List[str]]`, *optional*):
                The list of input images. We will replace the "<|image_i|>" in prompt with the 1-th image in list.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image. The number must be a multiple of 16.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image. The number must be a multiple of 16.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            use_img_guidance (`bool`, *optional*, defaults to True):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800). 
            img_guidance_scale (`float`, *optional*, defaults to 1.6):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800). 
            self.separate_cfg_infer (`bool`, *optional*, defaults to False):
                Perform inference on images with different guidance separately; this can save memory when generating images of large size at the expense of slower inference.
            self.use_kv_cache (`bool`, *optional*, defaults to True): enable kv cache to speed up the inference
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
        Examples:

        Returns:
            A list with the generated images.
        """
        assert height%16 == 0 and width%16 == 0
        if self.separate_cfg_infer:
            self.use_kv_cache = False
            # raise "Currently, don't support both self.use_kv_cache and self.separate_cfg_infer"
        if input_images is None:
            use_img_guidance = False
        if isinstance(prompt, str):
            prompt = [prompt]
            input_images = [input_images] if input_images is not None else None

        input_data = self.processor(prompt, input_images, height=height, width=width, use_img_cfg=use_img_guidance, separate_cfg_input=self.separate_cfg_infer)

        num_prompt = len(prompt)
        num_cfg = 2 if use_img_guidance else 1
        latent_size_h, latent_size_w = height//8, width//8

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=self.device, generator=generator)
        latents = torch.cat([latents]*(1+num_cfg), 0).to(self.dtype)

        input_img_latents = []
        if self.separate_cfg_infer:
            for temp_pixel_values in input_data['input_pixel_values']:
                temp_input_latents = []
                for img in temp_pixel_values:
                    img = self.vae_encode(img.to(self.device), self.dtype)
                    temp_input_latents.append(img)
                input_img_latents.append(temp_input_latents)
        else:
            for img in input_data['input_pixel_values']:
                img = self.vae_encode(img.to(self.device), self.dtype)
                input_img_latents.append(img)

        model_kwargs = dict(input_ids=self.move_to_device(input_data['input_ids']),
            input_img_latents=input_img_latents,
            input_image_sizes=input_data['input_image_sizes'],
            attention_mask=self.move_to_device(input_data["attention_mask"]),
            position_ids=self.move_to_device(input_data["position_ids"]),
            cfg_scale=guidance_scale,
            img_cfg_scale=img_guidance_scale,
            use_img_cfg=use_img_guidance,
            use_kv_cache=self.use_kv_cache)

        if self.separate_cfg_infer:
            func = self.model.forward_with_separate_cfg
        else:
            func = self.model.forward_with_cfg
        self.model.to(self.dtype)

        scheduler = OmniGenScheduler(num_steps=num_inference_steps)
        samples = scheduler(latents, func, model_kwargs, use_kv_cache=self.use_kv_cache)
        samples = samples.chunk((1+num_cfg), dim=0)[0]

        if output_type == 'latent':
            output_images = { 'images': samples }
            return output_images

        samples = samples.to(self.vae.dtype)
        if self.vae.config.shift_factor is not None:
            samples = samples / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            samples = samples / self.vae.config.scaling_factor
        samples = self.vae.decode(samples).sample

        output_samples = (samples * 0.5 + 0.5).clamp(0, 1)*255
        output_samples = output_samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        output_images = []
        for _i, sample in enumerate(output_samples):
            output_images.append(Image.fromarray(sample))

        return output_images
