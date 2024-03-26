import os
import json
import gradio as gr
import modules.shared
import modules.extensions


gradio_theme = gr.themes.Base()


def list_builtin_themes():
    files = [os.path.splitext(f)[0] for f in os.listdir('javascript') if f.endswith('.css') and f not in ['base.css', 'sdnext.css', 'style.css']]
    return files


def refresh_themes(no_update=False):
    fn = os.path.join('html', 'themes.json')
    res = []
    if os.path.exists(fn):
        try:
            with open(fn, 'r', encoding='utf8') as f:
                res = json.load(f)
        except Exception:
            modules.shared.log.error('Exception loading UI themes')
    if not no_update:
        try:
            modules.shared.log.info('Refreshing UI themes')
            r = modules.shared.req('https://huggingface.co/datasets/freddyaboulton/gradio-theme-subdomains/resolve/main/subdomains.json')
            if r.status_code == 200:
                res = r.json()
                modules.shared.writefile(res, fn)
            else:
                modules.shared.log.error('Error refreshing UI themes')
        except Exception:
            modules.shared.log.error('Exception refreshing UI themes')
    return res


def list_themes():
    builtin = list_builtin_themes()
    extensions = [e.name for e in modules.extensions.extensions if e.enabled]
    engines = []
    if 'sdnext-ui-ux' in extensions:
        ext = next((e for e in modules.extensions.extensions if e.name == 'sdnext-ui-ux'), None)
        folder = os.path.join(ext.path, 'themes')
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith('.css'):
                    engines.append(f'modern/{os.path.splitext(f)[0]}')
        if len(engines) == 0:
            engines.append('modern/sdxl_alpha')
    if 'sd-webui-lobe-theme' in extensions:
        modules.shared.log.info('Theme: installed="lobe"')
        engines.append('lobe')
    if 'Cozy-Nest' in extensions:
        modules.shared.log.info('Theme: installed="cozy-nest"')
        engines.append('cozy-nest')
    gradio = ["gradio/default", "gradio/base", "gradio/glass", "gradio/monochrome", "gradio/soft"]
    huggingface = refresh_themes(no_update=True)
    huggingface = {x['id'] for x in huggingface if x['status'] == 'RUNNING' and 'test' not in x['id'].lower()}
    huggingface = [f'huggingface/{x}' for x in huggingface]
    modules.shared.log.debug(f'Themes: builtin={len(builtin)} gradio={len(gradio)} huggingface={len(huggingface)}')
    themes = sorted(engines) + sorted(builtin) + sorted(gradio) + sorted(huggingface, key=str.casefold)
    return themes


def reload_gradio_theme(theme_name=None):
    global gradio_theme # pylint: disable=global-statement
    theme_name = theme_name or modules.shared.cmd_opts.theme or modules.shared.opts.gradio_theme
    if theme_name == 'default':
        theme_name = 'black-teal'
    if theme_name == 'modern':
        theme_name = 'modern/sdxl_alpha'
    modules.shared.opts.data['gradio_theme'] = theme_name
    default_font_params = {
        'font':['Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        'font_mono':['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace']
    }
    base = 'base.css'
    if theme_name.lower() in list_builtin_themes():
        base = 'sdnext.css'
        gradio_theme = gr.themes.Base(**default_font_params)
        modules.shared.log.info(f'UI theme: name="{theme_name}" style={modules.shared.opts.theme_style} base={base}')
        return True
    elif theme_name.lower() in ['lobe', 'cozy-nest']:
        gradio_theme = gr.themes.Base(**default_font_params)
        modules.shared.log.info(f'UI theme: name="{theme_name}" style={modules.shared.opts.theme_style} base={base}')
    elif theme_name.lower() == 'modern' or theme_name.lower().startswith('modern/'):
        gradio_theme = gr.themes.Base(**default_font_params)
        modules.shared.log.info(f'UI theme: name="{theme_name}" style={modules.shared.opts.theme_style} base={base}')
    elif theme_name.startswith("gradio/"):
        modules.shared.log.info(f'UI theme: name="{theme_name}" style={modules.shared.opts.theme_style} base={base}')
        modules.shared.log.warning('UI theme: using Gradio default theme which is not optimized for SD.Next')
        if theme_name == "gradio/default":
            gradio_theme = gr.themes.Default(**default_font_params)
        if theme_name == "gradio/base":
            gradio_theme = gr.themes.Base(**default_font_params)
        if theme_name == "gradio/glass":
            gradio_theme = gr.themes.Glass(**default_font_params)
        if theme_name == "gradio/monochrome":
            gradio_theme = gr.themes.Monochrome(**default_font_params)
        if theme_name == "gradio/soft":
            gradio_theme = gr.themes.Soft(**default_font_params)
    else:
        modules.shared.log.info(f'UI theme: name="{theme_name}" style={modules.shared.opts.theme_style} base={base}')
        try:
            hf_theme_name = theme_name.replace('huggingface/', '')
            modules.shared.log.warning('UI Theme: using 3rd party theme which is not optimized for SD.Next')
            gradio_theme = gr.themes.ThemeClass.from_hub(hf_theme_name)
        except Exception as e:
            modules.shared.log.error(f"UI theme: download error accessing HuggingFace {e}")
            gradio_theme = gr.themes.Default(**default_font_params)
    return False
