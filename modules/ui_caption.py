import gradio as gr
from modules import shared, ui_common, generation_parameters_copypaste
from modules.interrogate import openclip


def update_vlm_params(*args):
    vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample = args
    shared.opts.interrogate_vlm_max_length = vlm_max_tokens
    shared.opts.interrogate_vlm_num_beams = vlm_num_beams
    shared.opts.interrogate_vlm_temperature = vlm_temperature
    shared.opts.interrogate_vlm_do_sample = vlm_do_sample


def create_ui():
    with gr.Row(equal_height=False, variant='compact', elem_classes="caption"):
        with gr.Column(variant='compact'):
            with gr.Row():
                image = gr.Image(type='pil', label="Image")
            with gr.Tabs(elem_id="mode_caption"):
                with gr.Tab("CLiP Interrogate"):
                    with gr.Row():
                        clip_model = gr.Dropdown([], value=shared.opts.interrogate_clip_model, label='CLiP model')
                        ui_common.create_refresh_button(clip_model, openclip.refresh_clip_models, lambda: {"choices": openclip.refresh_clip_models()}, 'refresh_interrogate_models')
                        blip_model = gr.Dropdown(list(openclip.caption_models), value=shared.opts.interrogate_blip_model, label='Caption model')
                        mode = gr.Dropdown(openclip.caption_types, label='Mode', value='fast')
                    with gr.Accordion(label='Advanced', open=False, visible=True):
                        with gr.Row():
                            caption_max_length = gr.Slider(label='Max length', value=shared.opts.interrogate_clip_max_length, minimum=16, maximum=512, min_width=300)
                            chunk_size = gr.Slider(label='Chunk size', value=1024, minimum=256, maximum=4096, min_width=300)
                        with gr.Row():
                            min_flavors = gr.Slider(label='Min flavors', value=2, minimum=1, maximum=16, min_width=300)
                            max_flavors = gr.Slider(label='Max flavors', value=8, minimum=1, maximum=64, min_width=300)
                            flavor_intermediate_count = gr.Slider(label='Intermediates', value=1024, minimum=256, maximum=4096)
                        caption_max_length.change(fn=openclip.update_interrogate_params, inputs=[caption_max_length, chunk_size, min_flavors, max_flavors, flavor_intermediate_count], outputs=[])
                        chunk_size.change(fn=openclip.update_interrogate_params, inputs=[caption_max_length, chunk_size, min_flavors, max_flavors, flavor_intermediate_count], outputs=[])
                        min_flavors.change(fn=openclip.update_interrogate_params, inputs=[caption_max_length, chunk_size, min_flavors, max_flavors, flavor_intermediate_count], outputs=[])
                        max_flavors.change(fn=openclip.update_interrogate_params, inputs=[caption_max_length, chunk_size, min_flavors, max_flavors, flavor_intermediate_count], outputs=[])
                        flavor_intermediate_count.change(fn=openclip.update_interrogate_params, inputs=[caption_max_length, chunk_size, min_flavors, max_flavors, flavor_intermediate_count], outputs=[])
                    with gr.Accordion(label='Batch', open=False, visible=True):
                        with gr.Row():
                            batch_files = gr.File(label="Files", show_label=True, file_count='multiple', file_types=['image'], type='file', interactive=True, height=100)
                        with gr.Row():
                            batch_folder = gr.File(label="Folder", show_label=True, file_count='directory', file_types=['image'], type='file', interactive=True, height=100)
                        with gr.Row():
                            batch_str = gr.Text(label="Folder", value="", interactive=True)
                        with gr.Row():
                            batch = gr.Text(label="Prompts", lines=10)
                        with gr.Row():
                            clip_model = gr.Dropdown([], value='ViT-L-14/openai', label='CLiP Batch Model')
                            ui_common.create_refresh_button(clip_model, openclip.refresh_clip_models, lambda: {"choices": openclip.refresh_clip_models()}, 'refresh_interrogate_models')
                        with gr.Row(elem_id='interrogate_buttons_batch'):
                            btn_interrogate_batch = gr.Button("Batch interrogate", elem_id="interrogate_btn_interrogate", variant='primary')
                        with gr.Row():
                            save_output = gr.Checkbox(label='Save output', value=True, elem_id="extras_save_output")
                    with gr.Row(elem_id='interrogate_buttons_image'):
                        btn_interrogate_img = gr.Button("Interrogate", elem_id="interrogate_btn_interrogate", variant='primary')
                        btn_analyze_img = gr.Button("Analyze", elem_id="interrogate_btn_analyze", variant='primary')
                with gr.Tab("VLM Caption"):
                    from modules.interrogate import vqa
                    with gr.Row():
                        vqa_question = gr.Dropdown(label="Predefined question", allow_custom_value=False, choices=vqa.vlm_prompts, value=vqa.vlm_prompts[2])
                    with gr.Row():
                        vqa_prompt = gr.Textbox(label="Prompt", placeholder="optionally enter custom prompt", lines=2)
                    with gr.Row(elem_id='interrogate_buttons_query'):
                        vqa_model = gr.Dropdown(list(vqa.vlm_models), value=list(vqa.vlm_models)[0], label='VLM Model')
                    with gr.Accordion(label='Advanced', open=False, visible=True):
                        with gr.Row():
                            vlm_max_tokens = gr.Slider(label='Max tokens', value=shared.opts.interrogate_vlm_max_length, minimum=16, maximum=4096, step=1)
                            vlm_num_beams = gr.Slider(label='Num beams', value=shared.opts.interrogate_vlm_num_beams, minimum=1, maximum=16, step=1)
                            vlm_temperature = gr.Slider(label='Temperature', value=shared.opts.interrogate_vlm_temperature, minimum=0.1, maximum=1.0, step=0.01)
                        with gr.Row():
                            vlm_do_sample = gr.Checkbox(label='Use sample', value=shared.opts.interrogate_vlm_do_sample)
                        vlm_max_tokens.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample], outputs=[])
                        vlm_num_beams.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample], outputs=[])
                        vlm_temperature.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample], outputs=[])
                        vlm_do_sample.change(fn=update_vlm_params, inputs=[vlm_max_tokens, vlm_num_beams, vlm_temperature, vlm_do_sample], outputs=[])
                    with gr.Row(elem_id='interrogate_buttons_query'):
                        vqa_submit = gr.Button("Caption", elem_id="interrogate_btn_interrogate", variant='primary')
        with gr.Column(variant='compact'):
            with gr.Row():
                prompt = gr.Textbox(label="Answer", lines=8, placeholder="ai generated image description")
            with gr.Row(elem_id="interrogate_labels"):
                medium = gr.Label(elem_id="interrogate_label_medium", label="Medium", num_top_classes=5, visible=False)
                artist = gr.Label(elem_id="interrogate_label_artist", label="Artist", num_top_classes=5, visible=False)
                movement = gr.Label(elem_id="interrogate_label_movement", label="Movement", num_top_classes=5, visible=False)
                trending = gr.Label(elem_id="interrogate_label_trending", label="Trending", num_top_classes=5, visible=False)
                flavor = gr.Label(elem_id="interrogate_label_flavor", label="Flavor", num_top_classes=5, visible=False)
            with gr.Row(elem_id='copy_buttons_interrogate'):
                copy_interrogate_buttons = generation_parameters_copypaste.create_buttons(["txt2img", "img2img", "control", "extras"])

    btn_interrogate_img.click(openclip.interrogate_image, inputs=[image, clip_model, blip_model, mode], outputs=[prompt])
    btn_analyze_img.click(openclip.analyze_image, inputs=[image, clip_model, blip_model], outputs=[medium, artist, movement, trending, flavor])
    btn_interrogate_batch.click(fn=openclip.interrogate_batch, inputs=[batch_files, batch_folder, batch_str, clip_model, blip_model, mode, save_output], outputs=[batch])
    vqa_submit.click(vqa.interrogate, inputs=[vqa_question, vqa_prompt, image, vqa_model], outputs=[prompt])

    for tabname, button in copy_interrogate_buttons.items():
        generation_parameters_copypaste.register_paste_params_button(generation_parameters_copypaste.ParamBinding(paste_button=button, tabname=tabname, source_text_component=prompt, source_image_component=image,))
    generation_parameters_copypaste.add_paste_fields("caption", image, None)
