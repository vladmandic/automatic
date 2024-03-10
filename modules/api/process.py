from typing import Optional, List
from threading import Lock
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from fastapi.responses import JSONResponse
from modules.api.helpers import decode_base64_to_image, encode_pil_to_base64
from modules import errors, shared


processor = None # cached instance of processor
errors.install()


class ReqPreprocess(BaseModel):
    image: str = Field(title="Image", description="The base64 encoded image")
    model: str = Field(title="Model", description="The model to use for preprocessing")
    params: Optional[dict] = Field(default={}, title="Settings", description="Preprocessor settings")

class ResPreprocess(BaseModel):
    model: str = Field(default='', title="Model", description="The processor model used")
    image: str = Field(default='', title="Image", description="The processed image in base64 format")

class ReqMask(BaseModel):
    image: str = Field(title="Image", description="The base64 encoded image")
    type: str = Field(title="Mask type", description="Type of masking image to return")
    mask: Optional[str] = Field(title="Mask", description="If optional maks image is not provided auto-masking will be performed")
    model: Optional[str] = Field(title="Model", description="The model to use for preprocessing")
    params: Optional[dict] = Field(default={}, title="Settings", description="Preprocessor settings")

class ResMask(BaseModel):
    mask: str = Field(default='', title="Image", description="The processed image in base64 format")

class ItemPreprocess(BaseModel):
    name: str = Field(title="Name")
    params: dict = Field(title="Params")

class ItemMask(BaseModel):
    models: List[str] = Field(title="Models")
    colormaps: List[str] = Field(title="Color maps")
    params: dict = Field(title="Params")
    types: List[str] = Field(title="Types")


class APIProcess():
    def __init__(self, queue_lock: Lock):
        self.queue_lock = queue_lock

    def get_preprocess(self):
        from modules.control import processors
        items = []
        for k, v in processors.config.items():
            items.append(ItemPreprocess(name=k, params=v.get('params', {})))
        return items

    def post_preprocess(self, req: ReqPreprocess):
        global processor # pylint: disable=global-statement
        from modules.control import processors
        models = list(processors.config)
        if req.model not in models:
            return JSONResponse(status_code=400, content={"error": f"Processor model not found: id={req.model}"})
        image = decode_base64_to_image(req.image)
        if processor is None or processor.processor_id != req.model:
            with self.queue_lock:
                processor = processors.Processor(req.model)
        for k, v in req.params.items():
            if k not in processors.config[processor.processor_id]['params']:
                return JSONResponse(status_code=400, content={"error": f"Processor invalid parameter: id={req.model} {k}={v}"})
        shared.state.begin('api-preprocess', api=True)
        processed = processor(image, local_config=req.params)
        image = encode_pil_to_base64(processed)
        shared.state.end(api=False)
        return ResPreprocess(model=processor.processor_id, image=image)

    def get_mask(self):
        from modules import masking
        return ItemMask(models=list(masking.MODELS), colormaps=masking.COLORMAP, params=vars(masking.opts), types=masking.TYPES)

    def post_mask(self, req: ReqMask):
        from modules import masking
        if req.model:
            if req.model not in masking.MODELS:
                return JSONResponse(status_code=400, content={"error": f"Mask model not found: id={req.model}"})
            else:
                masking.init_model(req.model)
        if req.type not in masking.TYPES:
            return JSONResponse(status_code=400, content={"error": f"Mask type not found: id={req.type}"})
        image = decode_base64_to_image(req.image)
        mask = decode_base64_to_image(req.mask) if req.mask else None
        for k, v in req.params.items():
            if not hasattr(masking.opts, k):
                return JSONResponse(status_code=400, content={"error": f"Mask invalid parameter: {k}={v}"})
            else:
                setattr(masking.opts, k, v)
        shared.state.begin('api-mask', api=True)
        with self.queue_lock:
            processed = masking.run_mask(input_image=image, input_mask=mask, return_type=req.type)
        shared.state.end(api=False)
        if processed is None:
            return JSONResponse(status_code=400, content={"error": "Mask is none"})
        image = encode_pil_to_base64(processed)
        return ResMask(mask=image)
