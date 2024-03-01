from typing import Optional
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from modules.api.helpers import decode_base64_to_image, encode_pil_to_base64


processor = None # cached instance of processor


class ReqPreprocess(BaseModel):
    image: str = Field(default=None, title="Image", description="The base64 encoded image")
    model: str = Field(default=None, title="Model", description="The model to use for preprocessing")
    params: Optional[dict] = Field(default={}, title="Settings", description="Preprocessor settings")


class ResPreprocess(BaseModel):
    model: str = Field(default=None, title="Model", description="The processor model used")
    image: str = Field(default=None, title="Image", description="The processed image in base64 format")


def get_preprocess():
    from modules.control import processors
    p = {}
    for k, v in processors.config.items():
        p[k] = v.get('params')
    return JSONResponse(p)


def post_preprocess(req: ReqPreprocess):
    global processor # pylint: disable=global-statement
    from modules.control import processors
    models = list(processors.config)
    if req.model not in models:
        raise HTTPException(status_code=404, detail=f"Processor model not found: id={req.model}")
    image = decode_base64_to_image(req.image)
    if processor is None or processor.processor_id != req.model:
        processor = processors.Processor(req.model)
    for k, v in req.params.items():
        if k not in processors.config[processor.processor_id]['params']:
            raise HTTPException(status_code=400, detail=f"Processor invalid parameter: id={req.model} {k}={v}")
    processed = processor(image, local_config=req.params)
    image = encode_pil_to_base64(processed)
    return ResPreprocess(model=processor.processor_id, image=image)
