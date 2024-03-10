import torch
import transformers
from PIL import Image
from modules import shared, devices


processor = None
model = None
loaded: str = None
MODELS = {
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


def interrogate(vqa_question, vqa_image, vqa_model):
    vqa_model = MODELS.get(vqa_model, None)
    shared.log.debug(f'VQA: model="{vqa_model}" question="{vqa_question}" image={vqa_image}')
    if vqa_image is None:
        answer = 'no image provided'
    if vqa_model is None:
        answer = 'no model selected'
    if 'git' in vqa_model.lower():
        answer = git(vqa_question, vqa_image, vqa_model)
    if 'vilt' in vqa_model.lower():
        answer = vilt(vqa_question, vqa_image, vqa_model)
    if 'blip' in vqa_model.lower():
        answer = blip(vqa_question, vqa_image, vqa_model)
    if 'pix' in vqa_model.lower():
        answer = pix(vqa_question, vqa_image, vqa_model)
    if 'moondream2' in vqa_model.lower():
        answer = moondream(vqa_question, vqa_image, vqa_model)
    else:
        answer = 'unknown model'
    if model is not None:
        model.to(devices.cpu)
    devices.torch_gc()
    return answer
