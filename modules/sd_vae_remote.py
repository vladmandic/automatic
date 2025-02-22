import io
import time
import base64
import torch
import requests
from PIL import Image
from safetensors.torch import _tobytes


hf_endpoints = {
    'sd': 'https://lqmfdhmzmy4dw51z.us-east-1.aws.endpoints.huggingface.cloud',
    'sdxl': 'https://m5fxqwyk0r3uu79o.us-east-1.aws.endpoints.huggingface.cloud',
    'f1': 'https://zy1z7fzxpgtltg06.us-east-1.aws.endpoints.huggingface.cloud',
}


def remote_decode(latents: torch.Tensor, width: int = 0, height: int = 0, model_type: str = None) -> Image.Image:
    from modules import devices, shared, errors, modelloader
    images = []
    model_type = model_type or shared.sd_model_type
    url = hf_endpoints.get(model_type, None)
    if url is None:
        shared.log.error(f'Decode: type="remote" type={model_type} unsuppported')
        return images
    t0 = time.time()
    modelloader.hf_login()
    latents = latents.unsqueeze(0) if len(latents.shape) == 3 else latents
    for i in range(latents.shape[0]):
        try:
            latent = latents[i].detach().clone().to(device=devices.cpu, dtype=devices.dtype).unsqueeze(0)
            encoded = base64.b64encode(_tobytes(latent, "inputs")).decode("utf-8")
            params = {"shape": list(latent.shape), "dtype": str(latent.dtype).split(".", maxsplit=1)[-1]}
            if (model_type == 'f1') and (width > 0) and (height > 0):
                params['width'] = width
                params['height'] = height
            response = requests.post(
                url=url,
                json={"inputs": encoded, "parameters": params},
                headers={"Content-Type": "application/json", "Accept": "image/jpeg"},
                timeout=60,
            )
            if not response.ok:
                shared.log.error(f'Decode: type="remote" model={model_type} code={response.status_code} {response.json()}')
            else:
                images.append(Image.open(io.BytesIO(response.content)))
        except Exception as e:
            shared.log.error(f'Decode: type="remote" model={model_type} {e}')
            errors.display(e, 'VAE')
    t1 = time.time()
    shared.log.debug(f'Decode: type="remote" model={model_type} args={params} images={images} time={t1-t0:.3f}s')
    return images
