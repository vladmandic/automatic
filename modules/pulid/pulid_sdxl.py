import os
import cv2
import insightface
import numpy as np
import torch
import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from basicsr.utils import img2tensor, tensor2img
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from insightface.app import FaceAnalysis

from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from encoders_transformer import IDFormer
from attention_processor import AttnProcessor2_0 as AttnProcessor
from attention_processor import IDAttnProcessor2_0 as IDAttnProcessor


class StableDiffusionXLPuLIDPipeline:
    def __init__(self, pipe: StableDiffusionXLPipeline, device: torch.device, sampler=None, cache_dir=None):
        super().__init__()
        self.device = device
        self.pipe = pipe
        self.cache_dir = cache_dir
        self.hack_unet_attn_layers(self.pipe.unet)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.id_adapter = IDFormer().to(self.device)

        # preprocessors
        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)

        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device)
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std

        # antelopev2
        # snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        local_dir = os.path.join(self.cache_dir, 'pulid', 'models', 'antelopev2')
        _loc = snapshot_download('DIAMONIK7777/antelopev2', local_dir=local_dir)
        self.app = FaceAnalysis(
            name='antelopev2',
            root=os.path.join(self.cache_dir, 'pulid'),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model(os.path.join(local_dir, 'glintr100.onnx'))
        self.handler_ante.prepare(ctx_id=0)

        self.load_pretrain()

        # other configs
        self.debug_img_list = []

        # karras schedule related code, borrow from lllyasviel/Omost
        linear_start = 0.00085
        linear_end = 0.012
        timesteps = 1000
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, timesteps, dtype=torch.float64) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0), dtype=torch.float32)

        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.log_sigmas = self.sigmas.log()
        self.sigma_data = 1.0

        if sampler is not None:
            self.sampler = sampler

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def get_sigmas_karras(self, n, rho=7.0):
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1 / rho)
        max_inv_rho = self.sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def hack_unet_attn_layers(self, unet):
        id_adapter_attn_procs = {}
        for name, _ in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                hidden_size = None
            if cross_attention_dim is not None:
                id_adapter_attn_procs[name] = IDAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(unet.device)
            else:
                id_adapter_attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(id_adapter_attn_procs)
        self.id_adapter_attn_layers = nn.ModuleList(unet.attn_processors.values())

    def load_pretrain(self):
        ckpt_path = hf_hub_download('guozinan/PuLID', 'pulid_v1.1.safetensors', local_dir=os.path.join(self.cache_dir, 'pulid'))
        state_dict = load_file(ckpt_path)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1 :]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    def get_id_embedding(self, image_list):
        """
        Args:
            image in image_list: numpy rgb image, range [0, 255]
        """
        id_cond_list = []
        id_vit_hidden_list = []
        for _ii, image in enumerate(image_list):
            self.face_helper.clean_all()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # get antelopev2 embedding
            face_info = self.app.get(image_bgr)
            if len(face_info) > 0:
                face_info = sorted(
                    face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
                )[
                    -1
                ]  # only use the maximum face
                id_ante_embedding = face_info['embedding']
                self.debug_img_list.append(
                    image[
                        int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                        int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                    ]
                )
            else:
                id_ante_embedding = None

            # using facexlib to detect and align face
            self.face_helper.read_image(image_bgr)
            self.face_helper.get_face_landmarks_5(only_center_face=True)
            self.face_helper.align_warp_face()
            if len(self.face_helper.cropped_faces) == 0:
                raise RuntimeError('facexlib align face fail')
            align_face = self.face_helper.cropped_faces[0]
            # incase insightface didn't detect face
            if id_ante_embedding is None:
                id_ante_embedding = self.handler_ante.get_feat(align_face)

            id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device)
            if id_ante_embedding.ndim == 1:
                id_ante_embedding = id_ante_embedding.unsqueeze(0)

            # parsing
            input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0 # pylint: disable=redefined-builtin
            input = input.to(self.device)
            parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(input)
            # only keep the face features
            face_features_image = torch.where(bg, white_image, self.to_gray(input))
            self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

            # transform img before sending to eva-clip-vit
            face_features_image = resize(
                face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC
            )
            face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
            id_cond_vit, id_vit_hidden = self.clip_vision_model(
                face_features_image, return_all_features=False, return_hidden=True, shuffle=False
            )
            id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
            id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

            id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

            id_cond_list.append(id_cond)
            id_vit_hidden_list.append(id_vit_hidden)

        id_uncond = torch.zeros_like(id_cond_list[0])
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden_list[0])):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden_list[0][layer_idx]))

        id_cond = torch.stack(id_cond_list, dim=1)
        id_vit_hidden = id_vit_hidden_list[0]
        for i in range(1, len(image_list)):
            for j, x in enumerate(id_vit_hidden_list[i]):
                id_vit_hidden[j] = torch.cat([id_vit_hidden[j], x], dim=1)
        id_embedding = self.id_adapter(id_cond, id_vit_hidden)
        uncond_id_embedding = self.id_adapter(id_uncond, id_vit_hidden_uncond)

        # return id_embedding
        return uncond_id_embedding, id_embedding

    def set_progress_bar_config(self, bar_format: str = None, ncols: int = 80, colour: str = None):
        import functools
        from tqdm.auto import trange as trange_orig
        import pulid_sampling
        pulid_sampling.trange = functools.partial(trange_orig, bar_format=bar_format, ncols=ncols, colour=colour)

    def sample(self, x, sigma, **extra_args):
        x_ddim_space = x / (sigma[:, None, None, None] ** 2 + self.sigma_data**2) ** 0.5
        t = self.timestep(sigma)
        cfg_scale = extra_args['cfg_scale']
        eps_positive = self.pipe.unet(x_ddim_space, t, return_dict=False, **extra_args['positive'])[0]
        eps_negative = self.pipe.unet(x_ddim_space, t, return_dict=False, **extra_args['negative'])[0]
        noise_pred = eps_negative + cfg_scale * (eps_positive - eps_negative)
        latent = x - noise_pred * sigma[:, None, None, None]
        if self.callback_on_step_end is not None:
            self.step += 1
            self.callback_on_step_end(self.pipe, step=self.step, timestep=t, kwargs={ 'latents': latent })
        return latent

    def init_latent(self, seed, size, image, strength): # pylint: disable=unused-argument
        if image is not None and strength > 0:
            # TODO pulid img2img
            # input can be PIL.Image or np.ndarray so it needs to be converted to rgb tensor
            # image must be resized, encoded and noised according to denoising strength
            # see below for example from StableDiffusionXLImg2ImgPipeline
            latents = None
            """
            image = self.image_processor.preprocess(image)
            latents = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                add_noise,
            )
            """
            raise NotImplementedError(f'PuLID: task=img2img class={self.__class__.__name__} pipe={self.pipe.__class__.__name__} image={image} strength={strength}')
        else:
            # standard txt2img will full noise
            latents = torch.randn((size[0], 4, size[1] // 8, size[2] // 8), device="cpu", generator=torch.manual_seed(seed))
            latents = latents.to(dtype=self.pipe.unet.dtype, device=self.device)
        return latents

    def __call__(
        self,
        prompt: str='',
        negative_prompt: str='',
        width: int=1024,
        height: int=1024,
        guidance_scale: float=7.0,
        num_inference_steps: int=50,
        seed: int=-1,
        image: np.ndarray=None,
        mask_image: np.ndarray=None,
        strength: float=0.3,
        id_embedding=None,
        uncond_id_embedding=None,
        id_scale: float=1.0,
        callback_on_step_end=None,
    ):
        self.step = 0 # pylint: disable=attribute-defined-outside-init
        self.callback_on_step_end = callback_on_step_end # pylint: disable=attribute-defined-outside-init
        size = (1, height, width)
        # sigmas
        sigmas = self.get_sigmas_karras(num_inference_steps).to(self.device)

        # latents
        noise = self.init_latent(seed, size, image, strength)
        latents = noise * sigmas[0].to(noise)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        add_time_ids = list((size[1], size[2]) + (0, 0) + (size[1], size[2]))
        add_time_ids = torch.tensor([add_time_ids], dtype=self.pipe.unet.dtype, device=self.device)
        add_neg_time_ids = add_time_ids.clone()

        sampler_kwargs = dict(
            cfg_scale=guidance_scale,
            positive=dict(
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
                cross_attention_kwargs={'id_embedding': id_embedding, 'id_scale': id_scale},
            ),
            negative=dict(
                encoder_hidden_states=negative_prompt_embeds,
                added_cond_kwargs={"text_embeds": negative_pooled_prompt_embeds, "time_ids": add_neg_time_ids},
                cross_attention_kwargs={'id_embedding': uncond_id_embedding, 'id_scale': id_scale},
            ),
        )

        latents = self.sampler(self.sample, latents, sigmas, extra_args=sampler_kwargs, disable=False)
        latents = latents.to(dtype=self.pipe.vae.dtype, device=self.device) / self.pipe.vae.config.scaling_factor
        images = self.pipe.vae.decode(latents).sample
        images = self.pipe.image_processor.postprocess(images, output_type='pil')

        if mask_image is not None:
            # TODO: pulid inpaint
            # easiest inpaint is to use normal img2img and then combine output with input using mask
            # note that mask can be binary or grayscale (soft mask)
            raise NotImplementedError(f'PuLID: task=inpaint class={self.__class__.__name__} pipe={self.pipe.__class__.__name__} mask_image={mask_image}')

        return images


class StableDiffusionXLPuLIDPipelineImage(StableDiffusionXLPuLIDPipeline):
    def __init__(self, pipe: StableDiffusionXLPipeline, device: torch.device, sampler=None, cache_dir=None): # pylint: disable=useless-parent-delegation
        super().__init__(pipe, device, sampler, cache_dir)
        # we dont do anything special here, just having different class so task-type can be detected/assigned


class StableDiffusionXLPuLIDPipelineInpaint(StableDiffusionXLPuLIDPipeline):
    def __init__(self, pipe: StableDiffusionXLPipeline, device: torch.device, sampler=None, cache_dir=None): # pylint: disable=useless-parent-delegation
        super().__init__(pipe, device, sampler, cache_dir)
        # we dont do anything special here, just having different class so task-type can be detected/assigned
