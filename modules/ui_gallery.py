import os
from datetime import datetime
import gradio as gr
from PIL import Image
from modules import ui_symbols, ui_common, images
from modules.ui_components import ToolButton


def read_image(fn):
    if not os.path.isfile(fn):
        return [[], '', f'Image not found: {fn}']
    stat = os.stat(fn)
    image = Image.open(fn)
    image.already_saved_as = fn
    geninfo, _items = images.read_info_from_image(image)
    log = f'''
        <p>Image <b>{image.width} x {image.height}</b>
         | Format <b>{image.format}</b>
         | Mode <b>{image.mode}</b>
         | Size <b>{stat.st_size:,}</b>
         | Modified <b>{datetime.fromtimestamp(stat.st_mtime)}</b></p><br>
        '''
    return [[image], geninfo, geninfo, log]


def create_ui():
    with gr.Blocks() as tab:
        with gr.Row():
            sort_buttons = []
            sort_buttons.append(ToolButton(value=ui_symbols.sort_alpha_asc, show_label=False))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_alpha_dsc, show_label=False))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_size_asc, show_label=False))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_size_dsc, show_label=False))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_num_asc, show_label=False))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_num_dsc, show_label=False))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_time_asc, show_label=False))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_time_dsc, show_label=False))
            gr.Textbox(show_label=False, placeholder='Search', elem_id='tab-gallery-search')
            gr.HTML('', elem_id='tab-gallery-status')
            for btn in sort_buttons:
                btn.click(fn=None, _js='gallerySort', inputs=[btn], outputs=[])
        with gr.Row():
            with gr.Column():
                gr.HTML('', elem_id='tab-gallery-folders')
            with gr.Column():
                gr.HTML('', elem_id='tab-gallery-files')
            with gr.Column():
                btn_gallery_image = gr.Button('', elem_id='tab-gallery-send-image', visible=False, interactive=True)
                gallery_images, gen_info, html_info, _html_info_formatted, html_log = ui_common.create_output_panel("gallery")
                btn_gallery_image.click(fn=read_image, _js='gallerySendImage', inputs=[html_info], outputs=[gallery_images, html_info, gen_info, html_log])
    return [(tab, 'Gallery', 'tab-gallery')]
