# repo: https://huggingface.co/gokaygokay/Flux-Prompt-Enhance/discussions/2

import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
from modules import shared, scripts, devices


repo_id = "gokaygokay/Flux-Prompt-Enhance"
num_return_sequences = 5


class Script(scripts.Script):
    prompts = [['']]
    tokenizer: AutoTokenizer = None
    model: AutoModelForSeq2SeqLM = None
    prefix: str = "enhance prompt: "
    button: gr.Button = None
    max_length: gr.Slider = None
    temperature: gr.Slider = None
    repetition_penalty: gr.Slider = None
    table: gr.DataFrame = None
    prompt: gr.Textbox = None

    def title(self):
        return 'Prompt enhance'

    def show(self, is_img2img):
        return shared.native

    def load(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('gokaygokay/Flux-Prompt-Enhance', cache_dir=shared.opts.diffusers_dir)
        if self.model is None:
            shared.log.info(f'Prompt enhance: model="{repo_id}"')
            self.model = AutoModelForSeq2SeqLM.from_pretrained('gokaygokay/Flux-Prompt-Enhance', cache_dir=shared.opts.diffusers_dir).to(device=devices.cpu, dtype=devices.dtype)

    def enhance(self, prompt, temperature, repetition_penalty, max_length):
        self.load()
        t0 = time.time()
        input_text = self.prefix + prompt
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(devices.device)
        self.model = self.model.to(devices.device)
        kwargs = {
            'max_length': max_length,
            'num_return_sequences': num_return_sequences,
            'do_sample': True,
            'temperature': temperature,
            'repetition_penalty': repetition_penalty
        }
        try:
            outputs = self.model.generate(input_ids, **kwargs)
        except Exception as e:
            shared.log.error(f'Prompt enhance: error="{e}"')
            return [['']]
        self.model = self.model.to(devices.cpu)
        prompts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        prompts = [[p] for p in prompts]
        t1 = time.time()
        shared.log.info(f'Prompt enhance: temperature={temperature} repetition={repetition_penalty} length={max_length} sequences={num_return_sequences} time={t1-t0:.2f}s')
        return prompts

    def select(self, cell: gr.SelectData, _table):
        prompt = cell.value
        shared.log.info(f'Prompt enhance: prompt="{prompt}"')
        return prompt

    def ui(self, _is_img2img):
        with gr.Row():
            self.button = gr.Button(value='Enhance prompt')
        with gr.Row():
            self.max_length = gr.Slider(label='Length', minimum=64, maximum=512, step=1, value=128)
            self.temperature = gr.Slider(label='Temperature', minimum=0.1, maximum=2.0, step=0.05, value=0.7)
            self.repetition_penalty = gr.Slider(label='Penalty', minimum=0.1, maximum=2.0, step=0.05, value=1.2)
        with gr.Row():
            self.table = gr.DataFrame(self.prompts, label='', show_label=False, interactive=False, wrap=True, datatype="str", col_count=1, row_count=num_return_sequences, headers=['Prompts'])

        if self.prompt is not None:
            self.button.click(fn=self.enhance, inputs=[self.prompt, self.temperature, self.repetition_penalty, self.max_length], outputs=[self.table])
            self.table.select(fn=self.select, inputs=[self.table], outputs=[self.prompt])
        return []

    def after_component(self, component, **kwargs): # searching for actual ui prompt components
        if getattr(component, 'elem_id', '') in ['txt2img_prompt', 'img2img_prompt', 'control_prompt']:
            self.prompt = component
