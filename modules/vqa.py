import json
import torch
import transformers
import transformers.dynamic_module_utils
from PIL import Image
from modules import shared, devices, errors


processor = None
model = None
loaded: str = None
MODELS = {
    "MS Florence 2 Base": "microsoft/Florence-2-base", # 0.5GB
    "MS Florence 2 Large": "microsoft/Florence-2-large", # 1.5GB
    "MiaoshouAI PromptGen 1.5 Base": "MiaoshouAI/Florence-2-base-PromptGen-v1.5@c06a5f02cc6071a5d65ee5d294cf3732d3097540", # 1.1GB
    "MiaoshouAI PromptGen 1.5 Large": "MiaoshouAI/Florence-2-large-PromptGen-v1.5@c06a5f02cc6071a5d65ee5d294cf3732d3097540", # 3.3GB
    "CogFlorence 2.0 Large": "thwri/CogFlorence-2-Large-Freeze", # 1.6GB
    "CogFlorence 2.2 Large": "thwri/CogFlorence-2.2-Large", # 1.6GB
    "Moondream 2": "vikhyatk/moondream2", # 3.7GB
    "GIT TextCaps Base": "microsoft/git-base-textcaps", # 0.7GB
    "GIT VQA Base": "microsoft/git-base-vqav2", # 0.7GB
    "GIT VQA Large": "microsoft/git-large-vqav2", # 1.6GB
    "BLIP Base": "Salesforce/blip-vqa-base", # 1.5GB
    "BLIP Large": "Salesforce/blip-vqa-capfilt-large", # 1.5GB
    "ViLT Base": "dandelin/vilt-b32-finetuned-vqa", # 0.5GB
    "Pix Textcaps": "google/pix2struct-textcaps-base", # 1.1GB
}


def git(question: str, image: Image.Image, repo: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        model = transformers.GitForCausalLM.from_pretrained(repo)
        processor = transformers.GitProcessor.from_pretrained(repo)
        loaded = repo
    model.to(devices.device, devices.dtype)
    shared.log.debug(f'VQA: class={model.__class__.__name__} processor={processor.__class__} model={repo}')

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    git_dict = {}
    git_dict['pixel_values'] = pixel_values.to(devices.device, devices.dtype)
    if len(question) > 0:
        input_ids = processor(text=question, add_special_tokens=False).input_ids
        input_ids = [processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        git_dict['input_ids'] = input_ids.to(devices.device)
    with devices.inference_context():
        generated_ids = model.generate(**git_dict)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    shared.log.debug(f'VQA: response={response}')
    return response


def blip(question: str, image: Image.Image, repo: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        model = transformers.BlipForQuestionAnswering.from_pretrained(repo)
        processor = transformers.BlipProcessor.from_pretrained(repo)
        loaded = repo
    model.to(devices.device, devices.dtype)
    inputs = processor(image, question, return_tensors="pt")
    inputs = inputs.to(devices.device, devices.dtype)
    with devices.inference_context():
        outputs = model.generate(**inputs)
    response = processor.decode(outputs[0], skip_special_tokens=True)

    model.to(devices.cpu)
    shared.log.debug(f'VQA: response={response}')
    return response


def vilt(question: str, image: Image.Image, repo: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        model = transformers.ViltForQuestionAnswering.from_pretrained(repo)
        processor = transformers.ViltProcessor.from_pretrained(repo)
        loaded = repo
    model.to(devices.device)
    shared.log.debug(f'VQA: class={model.__class__.__name__} processor={processor.__class__} model={repo}')

    inputs = processor(image, question, return_tensors="pt")
    inputs = inputs.to(devices.device)
    with devices.inference_context():
        outputs = model(**inputs)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    response = model.config.id2label[idx]

    shared.log.debug(f'VQA: response={response}')
    return response


def pix(question: str, image: Image.Image, repo: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        model = transformers.Pix2StructForConditionalGeneration.from_pretrained(repo)
        processor = transformers.Pix2StructProcessor.from_pretrained(repo)
        loaded = repo
    model.to(devices.device)
    shared.log.debug(f'VQA: class={model.__class__.__name__} processor={processor.__class__} model={repo}')

    if len(question) > 0:
        inputs = processor(images=image, text=question, return_tensors="pt").to(devices.device)
    else:
        inputs = processor(images=image, return_tensors="pt").to(devices.device)
    with devices.inference_context():
        outputs = model.generate(**inputs)
    response = processor.decode(outputs[0], skip_special_tokens=True)

    shared.log.debug(f'VQA: response={response}')
    return response


def moondream(question: str, image: Image.Image, repo: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    if model is None or loaded != repo:
        model = transformers.AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True) # revision = "2024-03-05"
        processor = transformers.AutoTokenizer.from_pretrained(repo) # revision = "2024-03-05"
        loaded = repo
        model.eval()
    model.to(devices.device, devices.dtype)
    shared.log.debug(f'VQA: class={model.__class__.__name__} processor={processor.__class__} model={repo}')

    if len(question) < 2:
        question = "Describe the image."
    encoded = model.encode_image(image)
    with devices.inference_context():
        response = model.answer_question(encoded, question, processor)

    shared.log.debug(f'VQA: response="{response}"')
    return response


def florence(question: str, image: Image.Image, repo: str = None, revision: str = None):
    global processor, model, loaded # pylint: disable=global-statement
    _get_imports = transformers.dynamic_module_utils.get_imports
    def get_imports(f):
        R = _get_imports(f)
        if "flash_attn" in R:
            R.remove("flash_attn") # flash_attn is optional
        return R
    if model is None or loaded != repo:
        transformers.dynamic_module_utils.get_imports = get_imports
        model = transformers.AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, revision=revision)
        processor = transformers.AutoProcessor.from_pretrained(repo, trust_remote_code=True, revision=revision)
        transformers.dynamic_module_utils.get_imports = _get_imports
        loaded = repo
        model.eval()
    model.to(devices.device, devices.dtype)
    shared.log.debug(f'VQA: class={model.__class__.__name__} processor={processor.__class__} model={repo}')

    if question.startswith('<'):
        task = question.split('>', 1)[0] + '>'
    else:
        task = '<MORE_DETAILED_CAPTION>'
        question = task + question
    inputs = processor(text=question, images=image, return_tensors="pt")
    input_ids = inputs['input_ids'].to(devices.device)
    pixel_values = inputs['pixel_values'].to(devices.device, devices.dtype)
    with devices.inference_context():
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        response = processor.post_process_generation(generated_text, task="task", image_size=(image.width, image.height))

    if 'task' in response:
        response = response['task']
    if 'answer' in response:
        response = response['answer']
    if isinstance(response, dict):
        response = json.dumps(response)
    response = response.replace('\n', '').replace('\r', '').replace('\t', '').strip()
    shared.log.debug(f'VQA: task={task} response="{response}"')
    return response


def interrogate(vqa_question, vqa_image, vqa_model_req):
    try:
        vqa_model = MODELS.get(vqa_model_req, None)
        revision = None
        if '@' in vqa_model:
            vqa_model, revision = vqa_model.split('@')
        shared.log.debug(f'VQA: model="{vqa_model}" question="{vqa_question}" image={vqa_image}')
        if vqa_image is None:
            answer = 'no image provided'
            return answer
        if vqa_model_req is None:
            answer = 'no model selected'
            return answer
        if vqa_model is None:
            answer = f'unknown: model={vqa_model_req} available={MODELS.keys()}'
            return answer
        if 'git' in vqa_model.lower():
            answer = git(vqa_question, vqa_image, vqa_model)
        elif 'vilt' in vqa_model.lower():
            answer = vilt(vqa_question, vqa_image, vqa_model)
        elif 'blip' in vqa_model.lower():
            answer = blip(vqa_question, vqa_image, vqa_model)
        elif 'pix' in vqa_model.lower():
            answer = pix(vqa_question, vqa_image, vqa_model)
        elif 'moondream2' in vqa_model.lower():
            answer = moondream(vqa_question, vqa_image, vqa_model)
        elif 'florence' in vqa_model.lower():
            answer = florence(vqa_question, vqa_image, vqa_model, revision)
        else:
            answer = 'unknown model'
    except Exception as e:
        errors.display(e, 'VQA')
        answer = 'error'
    if model is not None:
        model.to(devices.cpu)
    devices.torch_gc()
    return answer
