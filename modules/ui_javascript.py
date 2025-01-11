import os
import gradio.routes
import gradio.utils
from modules import shared, theme
from modules.paths import script_path, data_path
import modules.scripts


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)
    return f'file={web_path}?{os.path.getmtime(fn)}'


def html_head():
    head = ''
    main = ['script.js']
    for js in main:
        script_js = os.path.join(script_path, "javascript", js)
        head += f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'
    added = []
    for script in modules.scripts.list_scripts("javascript", ".js"):
        if script.filename in main:
            continue
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'
        added.append(script.path)
    for script in modules.scripts.list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'
        added.append(script.path)
    added = [a.replace(script_path, '').replace('\\', '/') for a in added]
    # log.debug(f'Adding JS scripts: {added}')
    return head


def html_body():
    body = ''
    inline = ''
    if shared.opts.theme_style != 'Auto':
        inline += f"set_theme('{shared.opts.theme_style.lower()}');"
    body += f'<script type="text/javascript">{inline}</script>\n'
    return body


def html_css(css: str):
    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    head = ''
    if css is not None:
        head += stylesheet(os.path.join(script_path, 'javascript', css))
    for cssfile in modules.scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue
        head += stylesheet(cssfile)

    usercss = os.path.join(data_path, "user.css") if os.path.exists(os.path.join(data_path, "user.css")) else None
    if modules.shared.opts.theme_type == 'Standard':
        themecss = os.path.join(script_path, "javascript", f"{modules.shared.opts.gradio_theme}.css")
        if os.path.exists(themecss):
            head += stylesheet(themecss)
            modules.shared.log.debug(f'UI theme: css="{themecss}" base="{css}" user="{usercss}"')
        else:
            modules.shared.log.error(f'UI theme: css="{themecss}" not found')
    elif modules.shared.opts.theme_type == 'Modern':
        theme_folder = next((e.path for e in modules.extensions.extensions if e.name == 'sdnext-modernui'), None)
        themecss = os.path.join(theme_folder or '', 'themes', f'{modules.shared.opts.gradio_theme}.css')
        if os.path.exists(themecss):
            head += stylesheet(themecss)
            modules.shared.log.debug(f'UI theme: css="{themecss}" base="{css}" user="{usercss}"')
        else:
            modules.shared.log.error(f'UI theme: css="{themecss}" not found')
    if usercss is not None:
        head += stylesheet(usercss)
    return head


def reload_javascript():
    base_css = theme.reload_gradio_theme()
    head = html_head()
    css = html_css(base_css)
    body = html_body()
    title = '<title>SD.Next</title>'
    manifest = f'<link rel="manifest" href="{webpath(os.path.join(script_path, "html", "manifest.json"))}">'

    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'<head>', f'<head>{title}'.encode("utf8"))
        res.body = res.body.replace(b'</head>', f'{head}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</head>', f'{manifest}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}{body}</body>'.encode("utf8"))
        lines = res.body.decode("utf8").split('\n')
        for line in lines:
            if 'meta name="twitter:' in line:
                res.body = res.body.replace(line.encode("utf8"), b'')
            # if 'href="https://fonts.googleapis.com"' in line or 'href="https://fonts.gstatic.com"' in line:
            #     res.body = res.body.replace(line.encode("utf8"), b'')
            if 'iframeResizer.contentWindow.min.js' in line:
                res.body = res.body.replace(line.encode("utf8"), b'src="file=javascript/iframeResizer.min.js"')
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


def setup_ui_api(app):
    from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
    from typing import List

    class QuicksettingsHint(BaseModel): # pylint: disable=too-few-public-methods
        name: str = Field(title="Name of the quicksettings field")
        label: str = Field(title="Label of the quicksettings field")

    def quicksettings_hint():
        return [QuicksettingsHint(name=k, label=v.label) for k, v in shared.opts.data_labels.items()]

    app.add_api_route("/internal/quicksettings-hint", quicksettings_hint, methods=["GET"], response_model=List[QuicksettingsHint])
    app.add_api_route("/internal/ping", lambda: {}, methods=["GET"])


if not hasattr(shared, 'GradioTemplateResponseOriginal'):
    shared.GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse
