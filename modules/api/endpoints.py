from typing import Optional
from fastapi.exceptions import HTTPException
from modules import shared
from modules.api import models, helpers



def get_samplers():
    from modules import sd_samplers
    return [{"name": sampler[0], "aliases":sampler[2], "options":sampler[3]} for sampler in sd_samplers.all_samplers]

def get_sd_vaes():
    from modules.sd_vae import vae_dict
    return [{"model_name": x, "filename": vae_dict[x]} for x in vae_dict.keys()]

def get_upscalers():
    return [{"name": upscaler.name, "model_name": upscaler.scaler.model_name, "model_path": upscaler.data_path, "model_url": None, "scale": upscaler.scale} for upscaler in shared.sd_upscalers]

def get_sd_models():
    from modules import sd_models, sd_models_config
    return [{"title": x.title, "model_name": x.name, "filename": x.filename, "type": x.type, "hash": x.shorthash, "sha256": x.sha256, "config": sd_models_config.find_checkpoint_config_near_filename(x)} for x in sd_models.checkpoints_list.values()]

def get_hypernetworks():
    return [{"name": name, "path": shared.hypernetworks[name]} for name in shared.hypernetworks]

def get_detailers():
    return [{"name":x.name(), "cmd_dir": getattr(x, "cmd_dir", None)} for x in shared.detailers]

def get_prompt_styles():
    return [{ 'name': v.name, 'prompt': v.prompt, 'negative_prompt': v.negative_prompt, 'extra': v.extra, 'filename': v.filename, 'preview': v.preview} for v in shared.prompt_styles.styles.values()]

def get_embeddings():
    from modules import sd_hijack
    db = sd_hijack.model_hijack.embedding_db
    def convert_embedding(embedding):
        return {"step": embedding.step, "sd_checkpoint": embedding.sd_checkpoint, "sd_checkpoint_name": embedding.sd_checkpoint_name, "shape": embedding.shape, "vectors": embedding.vectors}

    def convert_embeddings(embeddings):
        return {embedding.name: convert_embedding(embedding) for embedding in embeddings.values()}

    return {"loaded": convert_embeddings(db.word_embeddings), "skipped": convert_embeddings(db.skipped_embeddings)}

def get_loras():
    from modules.lora import network, networks
    def create_lora_json(obj: network.NetworkOnDisk):
        return { "name": obj.name, "alias": obj.alias, "path": obj.filename, "metadata": obj.metadata }
    return [create_lora_json(obj) for obj in networks.available_networks.values()]

def get_extra_networks(page: Optional[str] = None, name: Optional[str] = None, filename: Optional[str] = None, title: Optional[str] = None, fullname: Optional[str] = None, hash: Optional[str] = None): # pylint: disable=redefined-builtin
    res = []
    for pg in shared.extra_networks:
        if page is not None and pg.name != page.lower():
            continue
        for item in pg.items:
            if name is not None and item.get('name', '') != name:
                continue
            if title is not None and item.get('title', '') != title:
                continue
            if filename is not None and item.get('filename', '') != filename:
                continue
            if fullname is not None and item.get('fullname', '') != fullname:
                continue
            if hash is not None and (item.get('shorthash', None) or item.get('hash')) != hash:
                continue
            res.append({
                'name': item.get('name', ''),
                'type': pg.name,
                'title': item.get('title', None),
                'fullname': item.get('fullname', None),
                'filename': item.get('filename', None),
                'hash': item.get('shorthash', None) or item.get('hash'),
                "preview": item.get('preview', None),
            })
    return res

def get_interrogate():
    from modules.interrogate import get_clip_models
    return ['clip', 'deepdanbooru'] + get_clip_models()

def post_interrogate(req: models.ReqInterrogate):
    if req.image is None or len(req.image) < 64:
        raise HTTPException(status_code=404, detail="Image not found")
    image = helpers.decode_base64_to_image(req.image)
    image = image.convert('RGB')
    if req.model == "clip":
        try:
            caption = shared.interrogator.interrogate(image)
        except Exception as e:
            caption = str(e)
        return models.ResInterrogate(caption=caption)
    elif req.model == "deepdanbooru" or req.model == 'deepbooru':
        from modules import deepbooru
        caption = deepbooru.model.tag(image)
        return models.ResInterrogate(caption=caption)
    else:
        from modules.interrogate import interrogate_image, analyze_image, get_clip_models
        if req.model not in get_clip_models():
            raise HTTPException(status_code=404, detail="Model not found")
        try:
            caption = interrogate_image(image, clip_model=req.clip_model, blip_model=req.blip_model, mode=req.mode)
        except Exception as e:
            caption = str(e)
        if not req.analyze:
            return models.ResInterrogate(caption=caption)
        else:
            medium, artist, movement, trending, flavor = analyze_image(image, clip_model=req.clip_model, blip_model=req.blip_model)
            return models.ResInterrogate(caption=caption, medium=medium, artist=artist, movement=movement, trending=trending, flavor=flavor)

def post_vqa(req: models.ReqVQA):
    if req.image is None or len(req.image) < 64:
        raise HTTPException(status_code=404, detail="Image not found")
    image = helpers.decode_base64_to_image(req.image)
    image = image.convert('RGB')
    from modules import vqa
    answer = vqa.interrogate(req.question, image, req.model)
    return models.ResVQA(answer=answer)

def post_unload_checkpoint():
    from modules import sd_models
    sd_models.unload_model_weights(op='model')
    sd_models.unload_model_weights(op='refiner')
    return {}

def post_reload_checkpoint():
    from modules import sd_models
    sd_models.reload_model_weights()
    return {}

def post_refresh_checkpoints():
    return shared.refresh_checkpoints()

def post_refresh_vae():
    return shared.refresh_vaes()

def post_refresh_loras():
    from modules.lora import networks
    return networks.list_available_networks()

def get_extensions_list():
    from modules import extensions
    extensions.list_extensions()
    ext_list = []
    for ext in extensions.extensions:
        ext: extensions.Extension
        ext.read_info()
        if ext.remote is not None:
            ext_list.append({
                "name": ext.name,
                "remote": ext.remote,
                "branch": ext.branch,
                "commit_hash":ext.commit_hash,
                "commit_date":ext.commit_date,
                "version":ext.version,
                "enabled":ext.enabled
            })
    return ext_list

def post_pnginfo(req: models.ReqImageInfo):
    from modules import images, script_callbacks, infotext
    if not req.image.strip():
        return models.ResImageInfo(info="")
    image = helpers.decode_base64_to_image(req.image.strip())
    if image is None:
        return models.ResImageInfo(info="")
    geninfo, items = images.read_info_from_image(image)
    if geninfo is None:
        geninfo = ""
    params = infotext.parse(geninfo)
    script_callbacks.infotext_pasted_callback(geninfo, params)
    return models.ResImageInfo(info=geninfo, items=items, parameters=params)

def get_history():
    return shared.history.list

def post_history(req: models.ReqHistory):
    shared.history.index = shared.history.find(req.name)
    return shared.history.index
