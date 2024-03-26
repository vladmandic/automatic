import json
import gradio as gr
from modules import scripts, shared, ui_common, postprocessing, call_queue, interrogate
import modules.generation_parameters_copypaste as parameters_copypaste
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call # pylint: disable=unused-import
from modules.extras import run_pnginfo
from modules.ui_common import infotext_to_html


def submit_info(image):
    _, geninfo, info = run_pnginfo(image)
    return infotext_to_html(geninfo), info, geninfo


def submit_process(tab_index, extras_image, image_batch, extras_batch_input_dir, extras_batch_output_dir, show_extras_results, save_output, *script_inputs):
    result_images, geninfo, js_info = postprocessing.run_postprocessing(tab_index, extras_image, image_batch, extras_batch_input_dir, extras_batch_output_dir, show_extras_results, *script_inputs, save_output=save_output)
    return result_images, geninfo, json.dumps(js_info), ''


def create_ui():
    tab_index = gr.State(value=0) # pylint: disable=abstract-class-instantiated
    with gr.Row(equal_height=False, variant='compact', elem_classes="extras"):
        with gr.Column(variant='compact'):
            with gr.Tabs(elem_id="mode_extras"):
                with gr.Tab('Process Image', id="single_image", elem_id="extras_single_tab") as tab_single:
                    with gr.Row():
                        extras_image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="extras_image")
                    with gr.Row(elem_id='copy_buttons_process'):
                        copy_process_buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "control"])
                with gr.Tab('Process Batch', id="batch_process", elem_id="extras_batch_process_tab") as tab_batch:
                    image_batch = gr.Files(label="Batch process", interactive=True, elem_id="extras_image_batch")
                with gr.Tab('Process Folder', id="batch_from_directory", elem_id="extras_batch_directory_tab") as tab_batch_dir:
                    extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir")
                    extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir")
                    show_extras_results = gr.Checkbox(label='Show result images', value=True, elem_id="extras_show_extras_results")

                with gr.Tab("Interrogate Image"):
                    with gr.Row():
                        image = gr.Image(type='pil', label="Image")
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", lines=3)
                    with gr.Row(elem_id="interrogate_labels"):
                        medium = gr.Label(elem_id="interrogate_label_medium", label="Medium", num_top_classes=5)
                        artist = gr.Label(elem_id="interrogate_label_artist", label="Artist", num_top_classes=5)
                        movement = gr.Label(elem_id="interrogate_label_movement", label="Movement", num_top_classes=5)
                        trending = gr.Label(elem_id="interrogate_label_trending", label="Trending", num_top_classes=5)
                        flavor = gr.Label(elem_id="interrogate_label_flavor", label="Flavor", num_top_classes=5)
                    with gr.Row():
                        clip_model = gr.Dropdown([], value='ViT-L-14/openai', label='CLIP Model')
                        ui_common.create_refresh_button(clip_model, interrogate.get_clip_models, lambda: {"choices": interrogate.get_clip_models()}, 'refresh_interrogate_models')
                        mode = gr.Radio(['best', 'fast', 'classic', 'caption', 'negative'], label='Mode', value='best')
                    with gr.Row(elem_id='interrogate_buttons_image'):
                        btn_interrogate_img = gr.Button("Interrogate", elem_id="interrogate_btn_interrogate", variant='primary')
                        btn_analyze_img = gr.Button("Analyze", elem_id="interrogate_btn_analyze", variant='primary')
                        btn_unload = gr.Button("Unload", elem_id="interrogate_btn_unload")
                    with gr.Row(elem_id='copy_buttons_interrogate'):
                        copy_interrogate_buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "extras", "control"])
                    btn_interrogate_img.click(interrogate.interrogate_image, inputs=[image, clip_model, mode], outputs=prompt)
                    btn_analyze_img.click(interrogate.analyze_image, inputs=[image, clip_model], outputs=[medium, artist, movement, trending, flavor])
                    btn_unload.click(interrogate.unload_clip_model)
                with gr.Tab("Interrogate Batch"):
                    with gr.Row():
                        batch_files = gr.File(label="Files", show_label=True, file_count='multiple', file_types=['image'], type='file', interactive=True, height=100)
                    with gr.Row():
                        batch_folder = gr.File(label="Folder", show_label=True, file_count='directory', file_types=['image'], type='file', interactive=True, height=100)
                    with gr.Row():
                        batch_str = gr.Text(label="Folder", value="", interactive=True)
                    with gr.Row():
                        batch = gr.Text(label="Prompts", lines=10)
                    with gr.Row():
                        clip_model = gr.Dropdown([], value='ViT-L-14/openai', label='CLIP Model')
                        ui_common.create_refresh_button(clip_model, interrogate.get_clip_models, lambda: {"choices": interrogate.get_clip_models()}, 'refresh_interrogate_models')
                    with gr.Row(elem_id='interrogate_buttons_batch'):
                        btn_interrogate_batch = gr.Button("Interrogate", elem_id="interrogate_btn_interrogate", variant='primary')
                with gr.Tab("Visual Query"):
                    from modules import vqa
                    with gr.Row():
                        vqa_image = gr.Image(type='pil', label="Image")
                    with gr.Row():
                        vqa_question = gr.Textbox(label="Question", placeholder="Descirbe the image")
                    with gr.Row():
                        vqa_answer = gr.Textbox(label="Answer", lines=3)
                    with gr.Row(elem_id='interrogate_buttons_query'):
                        vqa_model = gr.Dropdown(list(vqa.MODELS), value='Moondream 2', label='VQA Model')
                        vqa_submit = gr.Button("Interrogate", elem_id="interrogate_btn_interrogate", variant='primary')
                    vqa_submit.click(vqa.interrogate, inputs=[vqa_question, vqa_image, vqa_model], outputs=[vqa_answer])

                with gr.Row():
                    save_output = gr.Checkbox(label='Save output', value=True, elem_id="extras_save_output")

            script_inputs = scripts.scripts_postproc.setup_ui()
        with gr.Column():
            id_part = 'extras'
            with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
                interrupt = gr.Button('Stop', elem_id=f"{id_part}_interrupt", variant='secondary')
                interrupt.click(fn=lambda: shared.state.interrupt(), inputs=[], outputs=[])
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip", variant='secondary')
                skip.click(fn=lambda: shared.state.skip(), inputs=[], outputs=[])
            result_images, generation_info, html_info, html_info_formatted, html_log = ui_common.create_output_panel("extras")
            gr.HTML('File metadata')
            exif_info = gr.HTML(elem_id="pnginfo_html_info")
            gen_info = gr.Text(elem_id="pnginfo_gen_info", visible=False)
        for tabname, button in copy_process_buttons.items():
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=button, tabname=tabname, source_text_component=gen_info, source_image_component=extras_image))
        for tabname, button in copy_interrogate_buttons.items():
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=button, tabname=tabname, source_text_component=prompt, source_image_component=image,))


    tab_single.select(fn=lambda: 0, inputs=[], outputs=[tab_index])
    tab_batch.select(fn=lambda: 1, inputs=[], outputs=[tab_index])
    tab_batch_dir.select(fn=lambda: 2, inputs=[], outputs=[tab_index])
    extras_image.change(
        fn=wrap_gradio_call(submit_info),
        inputs=[extras_image],
        outputs=[html_info_formatted, exif_info, gen_info],
    )
    submit.click(
        _js="submit_postprocessing",
        fn=call_queue.wrap_gradio_gpu_call(submit_process, extra_outputs=[None, '']),
        inputs=[
            tab_index,
            extras_image,
            image_batch,
            extras_batch_input_dir,
            extras_batch_output_dir,
            show_extras_results,
            save_output,
            *script_inputs,
        ],
        outputs=[
            result_images,
            html_info,
            generation_info,
            html_log,
        ]
    )
    btn_interrogate_batch.click(
        fn=interrogate.interrogate_batch,
        inputs=[batch_files, batch_folder, batch_str, clip_model, mode, save_output],
        outputs=[batch],
    )

    parameters_copypaste.add_paste_fields("extras", extras_image, None)

    extras_image.change(
        fn=scripts.scripts_postproc.image_changed,
        inputs=[], outputs=[]
    )
