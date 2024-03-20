import os
import gradio as gr
from modules import shared, ui_symbols
from modules.ui_components import ToolButton


debug = shared.log.debug if os.environ.get('SD_GALLERY_DEBUG', None) is not None else lambda *args, **kwargs: None


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
                gr.HTML('', elem_id='tab-gallery-image')
    return [(tab, 'Gallery', 'tab-gallery')]
