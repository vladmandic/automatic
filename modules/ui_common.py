import json
import html
import os
import shutil
import platform
import subprocess
from functools import reduce
import gradio as gr
from modules import call_queue, shared, prompt_parser, ui_sections, ui_symbols, ui_components, generation_parameters_copypaste, images, scripts, script_callbacks, infotext


folder_symbol = ui_symbols.folder
debug = shared.log.trace if os.environ.get('SD_PASTE_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PASTE')


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def update_generation_info(generation_info, html_info, img_index):
    try:
        generation_info = json.loads(generation_info)
        if img_index < 0 or img_index >= len(generation_info["infotexts"]):
            return html_info, generation_info
        info = generation_info["infotexts"][img_index]
        html_info_formatted = infotext_to_html(info)
        return html_info, html_info_formatted
    except Exception:
        pass
    return html_info, html_info


def plaintext_to_html(text):
    res = '<p class="plaintext">' + "<br>\n".join([f"{html.escape(x)}" for x in text.split('\n')]) + '</p>'
    return res


def infotext_to_html(text):
    res = infotext.parse(text)
    prompt = res.get('Prompt', '')
    negative = res.get('Negative prompt', '')
    res.pop('Prompt', None)
    res.pop('Negative prompt', None)
    params = [f'{k}: {v}' for k, v in res.items() if v is not None and not k.endswith('-1') and not k.endswith('-2')]
    params = '| '.join(params) if len(params) > 0 else ''
    code = ''
    if len(prompt) > 0:
        code += f'<p><b>Prompt:</b> {html.escape(prompt)}</p>'
    if len(negative) > 0:
        code += f'<p><b>Negative:</b> {html.escape(negative)}</p>'
    if len(params) > 0:
        code += f'<p><b>Parameters:</b> {html.escape(params)}</p>'
    return code


def delete_files(js_data, files, _html_info, index):
    try:
        data = json.loads(js_data)
    except Exception:
        data = { 'index_of_first_image': 0 }
    start_index = 0
    if index > -1 and shared.opts.save_selected_only and (index >= data['index_of_first_image']):
        files = [files[index]]
        start_index = index
        filenames = []
    filenames = []
    fullfns = []
    for _image_index, filedata in enumerate(files, start_index):
        if 'name' in filedata and os.path.isfile(filedata['name']):
            fullfn = filedata['name']
            filenames.append(os.path.basename(fullfn))
            try:
                os.remove(fullfn)
                base, _ext = os.path.splitext(fullfn)
                desc = f'{base}.txt'
                if os.path.exists(desc):
                    os.remove(desc)
                fullfns.append(fullfn)
                shared.log.info(f"Deleting image: {fullfn}")
            except Exception as e:
                shared.log.error(f'Error deleting file: {fullfn} {e}')
    files = [image for image in files if image['name'] not in fullfns]
    return files, plaintext_to_html(f"Deleted: {filenames[0] if len(filenames) > 0 else 'none'}")


def save_files(js_data, files, html_info, index):
    os.makedirs(shared.opts.outdir_save, exist_ok=True)

    class PObject: # pylint: disable=too-few-public-methods
        def __init__(self, d=None):
            # 'Parser': 'Full parser', 'Model': 'lyriel_v16', 'Model hash': 'ec6f68ea63', 'Backend': 'Diffusers', 'App': 'SD.Next', 'Version': '69d7fef', 'Operations': 'txt2img', 'Sampler options': 'order 2/low order', 'Pipeline': 'StableDiffusionPipeline'
            # <pre>seq, uuid<br>date, datetime, job_timestamp<br>generation_number, batch_number<br>model, model_shortname<br>model_hash, model_name<br>sampler, seed, steps, cfg<br>clip_skip, denoising<br>hasprompt, prompt, styles<br>prompt_hash, prompt_no_styles<br>prompt_spaces, prompt_words<br>height, width, image_hash<br></pre>
            if d is not None:
                for k, v in d.items():
                    setattr(self, k, v)
            self.prompt = getattr(self, 'prompt', None) or getattr(self, 'Prompt', None) or ''
            self.negative_prompt = getattr(self, 'negative_prompt', None) or getattr(self, 'Negative_prompt', None) or ''
            self.sampler = getattr(self, 'sampler', None) or getattr(self, 'Sampler', None) or ''
            self.seed = getattr(self, 'seed', None) or getattr(self, 'Seed', None) or 0
            self.steps = getattr(self, 'steps', None) or getattr(self, 'Steps', None) or 0
            self.width = getattr(self, 'width', None) or getattr(self, 'Width', None) or getattr(self, 'Size-1', None) or 0
            self.height = getattr(self, 'height', None) or getattr(self, 'Height', None) or getattr(self, 'Size-2', None) or 0
            self.cfg_scale = getattr(self, 'cfg_scale', None) or getattr(self, 'CFG scale', None) or 0
            self.clip_skip = getattr(self, 'clip_skip', None) or getattr(self, 'Clip skip', None) or 1
            self.denoising_strength = getattr(self, 'denoising_strength', None) or getattr(self, 'Denoising', None) or 0
            self.index_of_first_image = getattr(self, 'index_of_first_image', 0)
            self.subseed = getattr(self, 'subseed', None) or getattr(self, 'Subseed', None)
            self.styles = getattr(self, 'styles', None) or getattr(self, 'Styles', None) or []
            self.styles = [s.strip() for s in self.styles.split(',')] if isinstance(self.styles, str) else self.styles

            self.outpath_grids = shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids
            self.infotexts = getattr(self, 'infotexts', [html_info])
            self.infotext = self.infotexts[0] if len(self.infotexts) > 0 else html_info
            self.all_negative_prompt = getattr(self, 'all_negative_prompts', [self.negative_prompt])
            self.all_prompts = getattr(self, 'all_prompts', [self.prompt])
            self.all_seeds = getattr(self, 'all_seeds', [self.seed])
            self.all_subseeds = getattr(self, 'all_subseeds', [self.subseed])
    try:
        data = json.loads(js_data)
    except Exception:
        data = {}
    p = PObject(data)
    start_index = 0
    if index > -1 and shared.opts.save_selected_only and (index >= p.index_of_first_image):  # ensures we are looking at a specific non-grid picture, and we have save_selected_only # pylint: disable=no-member
        files = [files[index]]
        start_index = index
    filenames = []
    fullfns = []
    for image_index, filedata in enumerate(files, start_index):
        is_grid = image_index < p.index_of_first_image # pylint: disable=no-member
        i = 0 if is_grid else (image_index - p.index_of_first_image) # pylint: disable=no-member
        while len(p.all_seeds) <= i:
            p.all_seeds.append(p.seed)
        while len(p.all_prompts) <= i:
            p.all_prompts.append(p.prompt)
        while len(p.infotexts) <= i:
            p.infotexts.append(p.infotext)
        if 'name' in filedata and ('tmp' not in filedata['name']) and os.path.isfile(filedata['name']):
            fullfn = filedata['name']
            filenames.append(os.path.basename(fullfn))
            fullfns.append(fullfn)
            destination = shared.opts.outdir_save
            namegen = images.FilenameGenerator(p, seed=p.all_seeds[i], prompt=p.all_prompts[i], image=None)  # pylint: disable=no-member
            dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]").lstrip(' ').rstrip('\\ /')
            destination = os.path.join(destination, dirname)
            destination = namegen.sanitize(destination)
            os.makedirs(destination, exist_ok = True)
            tgt_filename = os.path.join(destination, os.path.basename(fullfn))
            if not os.path.exists(tgt_filename):
                try:
                    shutil.copy(fullfn, destination)
                    shared.log.info(f'Copying image: file="{fullfn}" folder="{destination}"')
                except Exception as e:
                    shared.log.error(f'Copying image: {fullfn} {e}')
            if shared.opts.save_txt:
                try:
                    from PIL import Image
                    image = Image.open(fullfn)
                    info, _ = images.read_info_from_image(image)
                    filename_txt = f"{os.path.splitext(tgt_filename)[0]}.txt"
                    with open(filename_txt, "w", encoding="utf8") as file:
                        file.write(f"{info}\n")
                    shared.log.debug(f'Save: text="{filename_txt}"')
                except Exception as e:
                    shared.log.warning(f'Image description save failed: {filename_txt} {e}')
            script_callbacks.image_save_btn_callback(tgt_filename)
        else:
            image = generation_parameters_copypaste.image_from_url_text(filedata)
            info = p.infotexts[i + 1] if len(p.infotexts) > len(p.all_seeds) else p.infotexts[i] # infotexts may be offset by 1 because the first image is the grid
            if len(info) == 0:
                info = None
            if (js_data is None or len(js_data) == 0) and image is not None and image.info is not None:
                info = image.info.pop('parameters', None) or image.info.pop('UserComment', None)
                geninfo, _ = images.read_info_from_image(image)
                items = infotext.parse(geninfo)
                p = PObject(items)
            fullfn, txt_fullfn, _exif = images.save_image(image, shared.opts.outdir_save, "", seed=p.all_seeds[i], prompt=p.all_prompts[i], info=info, extension=shared.opts.samples_format, grid=is_grid, p=p)
            if fullfn is None:
                continue
            filename = os.path.relpath(fullfn, shared.opts.outdir_save)
            filenames.append(filename)
            fullfns.append(fullfn)
            if txt_fullfn:
                filenames.append(os.path.basename(txt_fullfn))
                # fullfns.append(txt_fullfn)
            script_callbacks.image_save_btn_callback(filename)
    if shared.opts.samples_save_zip and len(fullfns) > 1:
        zip_filepath = os.path.join(shared.opts.outdir_save, "images.zip")
        from zipfile import ZipFile
        with ZipFile(zip_filepath, "w") as zip_file:
            for i in range(len(fullfns)):
                if os.path.isfile(fullfns[i]):
                    with open(fullfns[i], mode="rb") as f:
                        zip_file.writestr(filenames[i], f.read())
        fullfns.insert(0, zip_filepath)
    return gr.File.update(value=fullfns, visible=True), plaintext_to_html(f"Saved: {filenames[0] if len(filenames) > 0 else 'none'}")


def open_folder(result_gallery, gallery_index = 0):
    try:
        folder = os.path.dirname(result_gallery[gallery_index]['name'])
    except Exception:
        folder = shared.opts.outdir_samples
    if not os.path.exists(folder):
        shared.log.warning(f'Folder open: folder="{folder}" does not exist')
        return
    elif not os.path.isdir(folder):
        shared.log.warning(f'Folder open: folder="{folder}" not a folder')
        return

    if not shared.cmd_opts.hide_ui_dir_config:
        path = os.path.normpath(folder)
        if platform.system() == "Windows":
            os.startfile(path) # pylint: disable=no-member
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path]) # pylint: disable=consider-using-with
        elif "microsoft-standard-WSL2" in platform.uname().release:
            subprocess.Popen(["wsl-open", path]) # pylint: disable=consider-using-with
        else:
            subprocess.Popen(["xdg-open", path]) # pylint: disable=consider-using-with


def interrogate_clip(image):
    if image is None:
        shared.log.error("Interrogate: no image selected")
        return gr.update()
    prompt = shared.interrogator.interrogate(image)
    return gr.update() if prompt is None else prompt


def interrogate_booru(image):
    if image is None:
        shared.log.error("Interrogate: no image selected")
        return gr.update()
    from modules import deepbooru
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


def create_output_panel(tabname, preview=True, prompt=None, height=None):
    with gr.Column(variant='panel', elem_id=f"{tabname}_results"):
        with gr.Group(elem_id=f"{tabname}_gallery_container"):
            if tabname == "txt2img":
                gr.HTML(value="", elem_id="main_info", visible=False, elem_classes=["main-info"])
            # columns are for <576px, <768px, <992px, <1200px, <1400px, >1400px
            result_gallery = gr.Gallery(value=[],
                                        label='Output',
                                        show_label=False,
                                        show_download_button=True,
                                        allow_preview=True,
                                        container=False,
                                        preview=preview,
                                        columns=4,
                                        object_fit='scale-down',
                                        height=height,
                                        elem_id=f"{tabname}_gallery",
                                        elem_classes=["gallery_main"],
                                       )
            if prompt is not None:
                interrogate_clip_btn, interrogate_booru_btn = ui_sections.create_interrogate_buttons('control')
                interrogate_clip_btn.click(fn=interrogate_clip, inputs=[result_gallery], outputs=[prompt])
                interrogate_booru_btn.click(fn=interrogate_booru, inputs=[result_gallery], outputs=[prompt])


        with gr.Column(elem_id=f"{tabname}_footer", elem_classes="gallery_footer"):
            dummy_component = gr.Label(visible=False)
            with gr.Row(elem_id=f"image_buttons_{tabname}", elem_classes="image-buttons"):
                if not shared.cmd_opts.listen:
                    open_folder_button = gr.Button('Show', visible=not shared.cmd_opts.hide_ui_dir_config, elem_id=f'open_folder_{tabname}')
                    open_folder_button.click(open_folder, _js="(gallery, dummy) => [gallery, selected_gallery_index()]", inputs=[result_gallery, dummy_component], outputs=[])
                else:
                    clip_files = gr.Button('Copy', elem_id=f'open_folder_{tabname}')
                    clip_files.click(fn=None, _js='clip_gallery_urls', inputs=[result_gallery], outputs=[])
                save = gr.Button('Save', elem_id=f'save_{tabname}')
                delete = gr.Button('Delete', elem_id=f'delete_{tabname}')
                if not shared.native:
                    buttons = generation_parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])
                else:
                    buttons = generation_parameters_copypaste.create_buttons(["txt2img", "img2img", "control", "extras"])

            download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False, elem_id=f'download_files_{tabname}')
            with gr.Group():
                html_info = gr.HTML(elem_id=f'html_info_{tabname}', elem_classes="infotext", visible=False) # contains raw infotext as returned by wrapped call
                html_info_formatted = gr.HTML(elem_id=f'html_info_formatted_{tabname}', elem_classes="infotext", visible=True) # contains html formatted infotext
                html_info.change(fn=infotext_to_html, inputs=[html_info], outputs=[html_info_formatted], show_progress=False)
                html_log = gr.HTML(elem_id=f'html_log_{tabname}')
                generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_{tabname}')
                generation_info_button = gr.Button(visible=False, elem_id=f"{tabname}_generation_info_button")

                generation_info_button.click(fn=update_generation_info, show_progress=False,
                    _js="(x, y, z) => [x, y, selected_gallery_index()]", # triggered on gallery change from js
                    inputs=[generation_info, html_info, html_info],
                    outputs=[html_info, html_info_formatted],
                )
                save.click(fn=call_queue.wrap_gradio_call(save_files), show_progress=False,
                    _js="(x, y, z, i) => [x, y, z, selected_gallery_index()]",
                    inputs=[generation_info, result_gallery, html_info, html_info],
                    outputs=[download_files, html_log],
                )
                delete.click(fn=call_queue.wrap_gradio_call(delete_files),show_progress=False,
                    _js="(x, y, z, i) => [x, y, z, selected_gallery_index()]",
                    inputs=[generation_info, result_gallery, html_info, html_info],
                    outputs=[result_gallery, html_log],
                )

            if tabname == "txt2img":
                paste_field_names = scripts.scripts_txt2img.paste_field_names
            elif tabname == "img2img":
                paste_field_names = scripts.scripts_img2img.paste_field_names
            elif tabname == "control":
                paste_field_names = scripts.scripts_control.paste_field_names
            else:
                paste_field_names = []
            debug(f'Paste field: tab={tabname} fields={paste_field_names}')
            for paste_tabname, paste_button in buttons.items():
                debug(f'Create output panel: source={tabname} target={paste_tabname} button={paste_button}')
                bindings = generation_parameters_copypaste.ParamBinding(
                    paste_button=paste_button,
                    tabname=paste_tabname,
                    source_tabname=tabname,
                    source_image_component=result_gallery,
                    paste_field_names=paste_field_names,
                    source_text_component=prompt or generation_info
                )
                generation_parameters_copypaste.register_paste_params_button(bindings)
            return result_gallery, generation_info, html_info, html_info_formatted, html_log


def create_refresh_button(refresh_component, refresh_method, refreshed_args = None, elem_id = None, visible: bool = True):
    def refresh():
        refresh_method()
        if refreshed_args is None:
            args = {"choices": refresh_method()} # pylint: disable=unnecessary-lambda-assignment
        elif callable(refreshed_args):
            args = refreshed_args()
        else:
            args = refreshed_args
        for k, v in args.items():
            setattr(refresh_component, k, v)
        return gr.update(**args)

    refresh_button = ui_components.ToolButton(value=ui_symbols.refresh, elem_id=elem_id, visible=visible)
    refresh_button.click(fn=refresh, inputs=[], outputs=[refresh_component])
    return refresh_button


def create_browse_button(browse_component, elem_id):
    def browse(folder):
        # import subprocess
        if folder is not None:
            return gr.update(value = folder)
        return gr.update()

    browse_button = ui_components.ToolButton(value=ui_symbols.folder, elem_id=elem_id)
    browse_button.click(fn=browse, _js="async () => await browseFolder()", inputs=[browse_component], outputs=[browse_component])
    # browse_button.click(fn=browse, inputs=[browse_component], outputs=[browse_component])
    return browse_button


def create_override_inputs(tab): # pylint: disable=unused-argument
    with gr.Row(elem_id=f"{tab}_override_settings_row"):
        override_settings = gr.Dropdown([], value=None, label="Override settings", visible=False, elem_id=f"{tab}_override_settings", multiselect=True)
        override_settings.change(fn=lambda x: gr.Dropdown.update(visible=len(x) > 0), inputs=[override_settings], outputs=[override_settings])
    return override_settings


def connect_reuse_seed(seed: gr.Number, reuse_seed: gr.Button, generation_info: gr.Textbox, is_subseed, subseed_strength=None):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""
    def copy_seed(gen_info_string: str, index: int):
        restore_seed = -1
        restore_strength = -1
        try:
            gen_info = json.loads(gen_info_string)
            shared.log.debug(f'Reuse: info={gen_info}')
            index -= gen_info.get('index_of_first_image', 0)
            index = int(index)
            if is_subseed:
                all_subseeds = gen_info.get('all_subseeds', [-1])
                restore_seed = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
                restore_strength = gen_info.get('subseed_strength', 0)
            else:
                all_seeds = gen_info.get('all_seeds', [-1])
                restore_seed = all_seeds[index if 0 <= index < len(all_seeds) else 0]
        except json.decoder.JSONDecodeError:
            if gen_info_string != '':
                shared.log.error(f"Error parsing JSON generation info: {gen_info_string}")
        if is_subseed is not None:
            return [restore_seed, gr_show(False), restore_strength]
        else:
            return [restore_seed, gr_show(False)]
    dummy_component = gr.Number(visible=False, value=0)
    if subseed_strength is None:
        reuse_seed.click(fn=copy_seed, _js="(x, y) => [x, selected_gallery_index()]", show_progress=False, inputs=[generation_info, dummy_component], outputs=[seed, dummy_component])
    else:
        reuse_seed.click(fn=copy_seed, _js="(x, y) => [x, selected_gallery_index()]", show_progress=False, inputs=[generation_info, dummy_component], outputs=[seed, dummy_component, subseed_strength])


def update_token_counter(text, steps):
    token_count = 0
    max_length = 75
    if shared.state.job_count > 0:
        shared.log.info('Tokenizer busy')
        return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"
    from modules import extra_networks
    prompt, _ = extra_networks.parse_prompt(text)
    if not shared.native:
        from modules import sd_hijack
        try:
            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
            prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)
        except Exception:
            prompt_schedules = [[[steps, text]]]
        flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
        prompts = [prompt_text for _step, prompt_text in flat_prompts]
        token_count, max_length = max([sd_hijack.model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    elif shared.native:
        if shared.sd_loaded and hasattr(shared.sd_model, 'tokenizer') and shared.sd_model.tokenizer is not None:
            has_bos_token = shared.sd_model.tokenizer.bos_token_id is not None
            has_eos_token = shared.sd_model.tokenizer.eos_token_id is not None
            ids = shared.sd_model.tokenizer(prompt)
            ids = getattr(ids, 'input_ids', [])
            token_count = len(ids) - int(has_bos_token) - int(has_eos_token)
            max_length = shared.sd_model.tokenizer.model_max_length - int(has_bos_token) - int(has_eos_token)
            if max_length is None or max_length < 0 or max_length > 10000:
                max_length = 0
    return gr.update(value=f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>", visible=token_count > 0)
