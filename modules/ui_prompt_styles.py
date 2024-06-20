# a1111 compatibility item, not used

import gradio as gr
from modules import shared, styles

styles_edit_symbol = '\U0001f58c\uFE0F'  # 🖌️
styles_materialize_symbol = '\U0001f4cb'  # 📋


def select_style(name):
    style = shared.prompt_styles.styles.get(name)
    existing = style is not None
    empty = not name
    prompt = style.prompt if style else gr.update()
    negative_prompt = style.negative_prompt if style else gr.update()
    return prompt, negative_prompt, gr.update(visible=existing), gr.update(visible=not empty)


def save_style(name, prompt, negative_prompt):
    if not name:
        return gr.update(visible=False)
    style = styles.Style(name, prompt, negative_prompt)
    shared.prompt_styles.styles[style.name] = style
    shared.prompt_styles.save_styles('')
    return gr.update(visible=True)


def delete_style(name):
    if name == "":
        return '', '', ''
    shared.prompt_styles.styles.pop(name, None)
    shared.prompt_styles.save_styles('')
    return '', '', ''


def materialize_styles(prompt, negative_prompt, styles): # pylint: disable=redefined-outer-name
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
    negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(negative_prompt, styles)
    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=negative_prompt), gr.Dropdown.update(value=[])]


def refresh_styles():
    return gr.update(choices=list(shared.prompt_styles.styles)), gr.update(choices=list(shared.prompt_styles.styles))


class UiPromptStyles:
    def __init__(self, tabname, main_ui_prompt, main_ui_negative_prompt): # pylint: disable=unused-argument
        self.dropdown = gr.Dropdown(label="Styles", elem_id=f"{tabname}_styles", choices=[style.name for style in shared.prompt_styles.styles.values()], value=[], multiselect=True)
