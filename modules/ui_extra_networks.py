import os
import io
import random
import re
import time
import json
import html
import base64
import urllib.parse
import threading
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
from html.parser import HTMLParser
from collections import OrderedDict
import gradio as gr
from PIL import Image
from starlette.responses import FileResponse, JSONResponse
from modules import paths, shared, files_cache, errors, infotext
from modules.ui_components import ToolButton
import modules.ui_symbols as symbols


allowed_dirs = []
refresh_time = 0
extra_pages = shared.extra_networks
debug = shared.log.trace if os.environ.get('SD_EN_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: EN')
card_full = '''
    <div class='card' onclick={card_click} title='{name}' data-tab='{tabname}' data-page='{page}' data-name='{name}' data-filename='{filename}' data-tags='{tags}' data-mtime='{mtime}' data-size='{size}' data-search='{search}' style='--data-color: {color}'>
        <div class='overlay'>
            <div class='tags'></div>
            <div class='name'>{title}</div>
        </div>
        <div class='version'>{version}</div>
        <div class='actions'>
            <span class='details' title="Get details" onclick="showCardDetails(event)">&#x1f6c8;</span>
            <div class='additional'><ul></ul></div>
        </div>
        <img class='preview' src='{preview}' style='width: {width}px; height: {height}px; object-fit: {fit}' loading='lazy'></img>
    </div>
'''
card_list = '''
    <div class='card card-list' onclick={card_click} title='{name}' data-tab='{tabname}' data-page='{page}' data-name='{name}' data-filename='{filename}' data-tags='{tags}' data-mtime='{mtime}' data-version='{version}' data-size='{size}' data-search='{search}'>
        <div style='display: flex'>
            <span class='details' title="Get details" onclick="showCardDetails(event)">&#x1f6c8;</span>&nbsp;
            <div class='name' style='flex-flow: column'>{title}&nbsp;
                <div class='tags tags-list'></div>
            </div>
        </div>
    </div>
'''
preview_map = None


def init_api(app):

    def fetch_file(filename: str = ""):
        if not os.path.exists(filename):
            return JSONResponse({ "error": f"file {filename}: not found" }, status_code=404)
        if filename.startswith('html/') or filename.startswith('models/'):
            return FileResponse(filename, headers={"Accept-Ranges": "bytes"})
        if not any(Path(folder).absolute() in Path(filename).absolute().parents for folder in allowed_dirs):
            return JSONResponse({ "error": f"file {filename}: must be in one of allowed directories" }, status_code=403)
        if os.path.splitext(filename)[1].lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            return JSONResponse({"error": f"file {filename}: not an image file"}, status_code=403)
        return FileResponse(filename, headers={"Accept-Ranges": "bytes"})

    def get_metadata(page: str = "", item: str = ""):
        page = next(iter([x for x in shared.extra_networks if x.name == page]), None)
        if page is None:
            return JSONResponse({ 'metadata': 'none' })
        metadata = page.metadata.get(item, 'none')
        if metadata is None:
            metadata = ''
        # shared.log.debug(f"Networks metadata: page='{page}' item={item} len={len(metadata)}")
        return JSONResponse({"metadata": metadata})

    def get_info(page: str = "", item: str = ""):
        page = next(iter([x for x in get_pages() if x.name == page]), None)
        if page is None:
            return JSONResponse({ 'info': 'none' })
        item = next(iter([x for x in page.items if x['name'] == item]), None)
        if item is None:
            return JSONResponse({ 'info': 'none' })
        info = page.find_info(item.get('filename', None) or item.get('name', None))
        if info is None:
            info = {}
        # shared.log.debug(f"Networks info: page='{page.name}' item={item['name']} len={len(info)}")
        return JSONResponse({"info": info})

    def get_desc(page: str = "", item: str = ""):
        page = next(iter([x for x in get_pages() if x.name == page]), None)
        if page is None:
            return JSONResponse({ 'description': 'none' })
        item = next(iter([x for x in page.items if x['name'] == item]), None)
        if item is None:
            return JSONResponse({ 'description': 'none' })
        desc = page.find_description(item.get('filename', None) or item.get('name', None))
        if desc is None:
            desc = ''
        # shared.log.debug(f"Networks desc: page='{page.name}' item={item['name']} len={len(desc)}")
        return JSONResponse({"description": desc})

    app.add_api_route("/sd_extra_networks/thumb", fetch_file, methods=["GET"])
    app.add_api_route("/sd_extra_networks/metadata", get_metadata, methods=["GET"])
    app.add_api_route("/sd_extra_networks/info", get_info, methods=["GET"])
    app.add_api_route("/sd_extra_networks/description", get_desc, methods=["GET"])


class ExtraNetworksPage:
    def __init__(self, title):
        self.title = title
        self.name = title.lower()
        self.allow_negative_prompt = False
        self.metadata = {}
        self.info = {}
        self.html = ''
        self.items = []
        self.missing_thumbs = []
        self.refresh_time = 0
        self.page_time = 0
        self.list_time = 0
        self.info_time = 0
        self.desc_time = 0
        self.preview_time = 0
        self.dirs = {}
        self.view = shared.opts.extra_networks_view
        self.card = card_full if shared.opts.extra_networks_view == 'gallery' else card_list

    def refresh(self):
        pass

    def patch(self, text: str, tabname: str):
        return text.replace('~tabname', tabname)

    def create_xyz_grid(self):
        """
        xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module

        def add_prompt(p, opt, x):
            for item in [x for x in self.items if x["name"] == opt]:
                try:
                    p.prompt = f'{p.prompt} {eval(item["prompt"])}' # pylint: disable=eval-used
                except Exception as e:
                    shared.log.error(f'Cannot evaluate extra network prompt: {item["prompt"]} {e}')

        if not any(self.title in x.label for x in xyz_grid.axis_options):
            if self.title == 'Model':
                return
            opt = xyz_grid.AxisOption(f"[Network] {self.title}", str, add_prompt, choices=lambda: [x["name"] for x in self.items])
            if opt not in xyz_grid.axis_options:
                xyz_grid.axis_options.append(opt)
        """

    def link_preview(self, filename):
        quoted_filename = urllib.parse.quote(filename.replace('\\', '/'))
        mtime = os.path.getmtime(filename) if os.path.exists(filename) else 0
        preview = f"./sd_extra_networks/thumb?filename={quoted_filename}&mtime={mtime}"
        return preview

    def is_empty(self, folder):
        return any(files_cache.list_files(folder, ext_filter=['.ckpt', '.safetensors', '.pt', '.json']))

    def create_thumb(self):
        debug(f'EN create-thumb: {self.name}')
        created = 0
        for f in self.missing_thumbs:
            if os.path.join('models', 'Reference') in f or not os.path.exists(f):
                continue
            fn = os.path.splitext(f)[0].replace('.preview', '')
            fn = f'{fn}.thumb.jpg'
            if os.path.exists(fn): # thumbnail already exists
                continue
            img = None
            try:
                img = Image.open(f)
            except Exception:
                img = None
                shared.log.warning(f'Extra network removing invalid image: {f}')
            try:
                if img is None:
                    img = None
                    os.remove(f)
                elif img.width > 1024 or img.height > 1024 or os.path.getsize(f) > 65536:
                    img = img.convert('RGB')
                    img.thumbnail((512, 512), Image.Resampling.HAMMING)
                    img.save(fn, quality=50)
                    img.close()
                    created += 1
            except Exception as e:
                shared.log.warning(f'Extra network error creating thumbnail: {f} {e}')
        if created > 0:
            shared.log.info(f"Network thumbnails: {self.name} created={created}")
            self.missing_thumbs.clear()

    def create_items(self, tabname):
        if self.refresh_time is not None and self.refresh_time > refresh_time: # cached results
            return
        t0 = time.time()
        try:
            self.items = list(self.list_items())
            self.refresh_time = time.time()
        except Exception as e:
            self.items = []
            shared.log.error(f'Networks: listing items class={self.__class__.__name__} tab={tabname} {e}')
            if os.environ.get('SD_EN_DEBUG', None):
                errors.display(e, f'Networks: listing items: class={self.__class__.__name__} tab={tabname}')
        for item in self.items:
            if item is None:
                continue
            self.metadata[item["name"]] = item.get("metadata", {})
        t1 = time.time()
        debug(f'EN create-items: page={self.name} items={len(self.items)} time={t1-t0:.2f}')
        self.list_time += t1-t0


    def create_page(self, tabname, skip = False):
        debug(f'EN create-page: {self.name}')
        if self.page_time > refresh_time and len(self.html) > 0: # cached page
            return self.patch(self.html, tabname)
        self_name_id = self.name.replace(" ", "_")
        if skip:
            return f"<div id='{tabname}_{self_name_id}_subdirs' class='extra-network-subdirs'></div><div id='{tabname}_{self_name_id}_cards' class='extra-network-cards'>Extra network page not ready<br>Click refresh to try again</div>"
        subdirs = {}
        allowed_folders = [os.path.abspath(x) for x in self.allowed_directories_for_previews() if os.path.exists(x)]
        for parentdir, dirs in {d: files_cache.walk(d, cached=True, recurse=files_cache.not_hidden) for d in allowed_folders}.items():
            for tgt in dirs:
                tgt = tgt.path
                if os.path.join(paths.models_path, 'Reference') in tgt:
                    subdirs['Reference'] = 1
                if shared.native and shared.opts.diffusers_dir in tgt:
                    subdirs[os.path.basename(shared.opts.diffusers_dir)] = 1
                if 'models--' in tgt:
                    continue
                subdir = tgt[len(parentdir):].replace("\\", "/")
                while subdir.startswith("/"):
                    subdir = subdir[1:]
                if not subdir:
                    continue
                # if not self.is_empty(tgt):
                subdirs[subdir] = 1
        debug(f"Networks: page='{self.name}' subfolders={list(subdirs)}")
        subdirs = OrderedDict(sorted(subdirs.items()))
        if self.name == 'model':
            subdirs['Reference'] = 1
            subdirs[os.path.basename(shared.opts.diffusers_dir)] = 1
            subdirs.move_to_end(os.path.basename(shared.opts.diffusers_dir))
            subdirs.move_to_end('Reference')
        if self.name == 'style' and shared.opts.extra_networks_styles:
            subdirs['built-in'] = 1
        subdirs_html = "<button class='lg secondary gradio-button custom-button search-all' onclick='extraNetworksSearchButton(event)'>All</button><br>"
        subdirs_html += "".join([f"<button class='lg secondary gradio-button custom-button' onclick='extraNetworksSearchButton(event)'>{html.escape(subdir)}</button><br>" for subdir in subdirs if subdir != ''])
        self.html = ''
        self.create_items(tabname)
        self.create_xyz_grid()
        htmls = []

        if len(self.items) > 0 and self.items[0].get('mtime', None) is not None:
            if shared.opts.extra_networks_sort == 'Default':
                pass
            elif shared.opts.extra_networks_sort == 'Name [A-Z]':
                self.items.sort(key=lambda x: x["name"])
            elif shared.opts.extra_networks_sort == 'Name [Z-A]':
                self.items.sort(key=lambda x: x["name"], reverse=True)
            elif shared.opts.extra_networks_sort == 'Date [Newest]':
                self.items.sort(key=lambda x: x["mtime"], reverse=True)
            elif shared.opts.extra_networks_sort == 'Date [Oldest]':
                self.items.sort(key=lambda x: x["mtime"])
            elif shared.opts.extra_networks_sort == 'Size [Largest]':
                self.items.sort(key=lambda x: x["size"], reverse=True)
            elif shared.opts.extra_networks_sort == 'Size [Smallest]':
                self.items.sort(key=lambda x: x["size"])

        for item in self.items:
            htmls.append(self.create_html(item, tabname))
        self.html += ''.join(htmls)
        self.page_time = time.time()
        self.html = f"<div id='~tabname_{self_name_id}_subdirs' class='extra-network-subdirs'>{subdirs_html}</div><div id='~tabname_{self_name_id}_cards' class='extra-network-cards'>{self.html}</div>"
        shared.log.debug(f"Networks: page='{self.name}' items={len(self.items)} subfolders={len(subdirs)} tab={tabname} folders={self.allowed_directories_for_previews()} list={self.list_time:.2f} thumb={self.preview_time:.2f} desc={self.desc_time:.2f} info={self.info_time:.2f} workers={shared.max_workers}")
        if len(self.missing_thumbs) > 0:
            threading.Thread(target=self.create_thumb).start()
        return self.patch(self.html, tabname)

    def list_items(self):
        raise NotImplementedError

    def allowed_directories_for_previews(self):
        return []

    def create_html(self, item, tabname):
        def random_bright_color():
            r = random.randint(100, 255)
            g = random.randint(100, 255)
            b = random.randint(100, 255)
            return '#{:02x}{:02x}{:02x}'.format(r, g, b) # pylint: disable=consider-using-f-string

        try:
            args = {
                "tabname": tabname,
                "page": self.name,
                "name": item.get('name', ''),
                "title": os.path.basename(item["name"].replace('_', ' ')),
                "filename": item.get('filename', ''),
                "tags": '|'.join([item.get('tags')] if isinstance(item.get('tags', {}), str) else list(item.get('tags', {}).keys())),
                "preview": html.escape(item.get('preview', None) or self.link_preview('html/card-no-preview.png')),
                "width": shared.opts.extra_networks_card_size,
                "height": shared.opts.extra_networks_card_size if shared.opts.extra_networks_card_square else 'auto',
                "fit": shared.opts.extra_networks_card_fit,
                "prompt": item.get("prompt", None),
                "search": item.get("search_term", ""),
                "description": item.get("description") or "",
                "card_click": item.get("onclick", '"' + html.escape(f'return cardClicked({item.get("prompt", None)}, {"true" if self.allow_negative_prompt else "false"})') + '"'),
                "mtime": item.get("mtime", 0),
                "size": item.get("size", 0),
                "version": item.get("version", ''),
                "color": random_bright_color(),
            }
            alias = item.get("alias", None)
            if alias is not None:
                args['title'] += f'\nAlias: {alias}'
            return self.card.format(**args)
        except Exception as e:
            shared.log.error(f'Networks: item error: page={tabname} item={item["name"]} {e}')
            if os.environ.get('SD_EN_DEBUG', None) is not None:
                errors.display(e, 'Networks')
            return ""

    def find_preview_file(self, path):
        if path is None:
            return 'html/card-no-preview.png'
        if os.path.join('models', 'Reference') in path:
            return path
        exts = ["jpg", "jpeg", "png", "webp", "tiff", "jp2"]
        reference_path = os.path.abspath(os.path.join('models', 'Reference'))
        files = list(files_cache.list_files(reference_path, ext_filter=exts, recursive=False))
        if shared.opts.diffusers_dir in path:
            path = os.path.relpath(path, shared.opts.diffusers_dir)
            fn = os.path.join(reference_path, path.replace('models--', '').replace('\\', '/').split('/')[0])
        else:
            fn = os.path.splitext(path)[0]
            files += list(files_cache.list_files(os.path.dirname(path), ext_filter=exts, recursive=False))
        for file in [f'{fn}{mid}{ext}' for ext in exts for mid in ['.thumb.', '.', '.preview.']]:
            if file in files:
                if '.thumb.' not in file:
                    self.missing_thumbs.append(file)
                return file
        return 'html/card-no-preview.png'

    def find_preview(self, filename):
        t0 = time.time()
        preview_file = self.find_preview_file(filename)
        self.preview_time += time.time() - t0
        return self.link_preview(preview_file)

    def update_all_previews(self, items):
        global preview_map # pylint: disable=global-statement
        if preview_map is None:
            preview_map = shared.readfile('html/previews.json', silent=True)
        t0 = time.time()
        reference_path = os.path.abspath(os.path.join('models', 'Reference'))
        possible_paths = list(set([os.path.dirname(item['filename']) for item in items] + [reference_path]))
        exts = ["jpg", "jpeg", "png", "webp", "tiff", "jp2"]
        all_previews = list(files_cache.list_files(*possible_paths, ext_filter=exts, recursive=False))
        all_previews_fn = [os.path.basename(x) for x in all_previews]
        for item in items:
            if item.get('preview', None) is not None:
                continue
            base = os.path.splitext(item['filename'])[0]
            if item.get('local_preview', None) is None:
                item['local_preview'] = f'{base}.{shared.opts.samples_format}'
            if shared.opts.diffusers_dir in base:
                match = re.search(r"models--([^/^\\]+)[/\\]", base)
                if match is None:
                    match = re.search(r"models--(.*)", base)
                base = os.path.join(reference_path, match[1])
                model_path = os.path.join(shared.opts.diffusers_dir, match[0])
                item['local_preview'] = f'{os.path.join(model_path, match[1])}.{shared.opts.samples_format}'
                all_previews += list(files_cache.list_files(model_path, ext_filter=exts, recursive=False))
            base = os.path.basename(base)
            for file in [f'{base}{mid}{ext}' for ext in exts for mid in ['.thumb.', '.', '.preview.']]:
                if file in all_previews_fn:
                    file_idx = all_previews_fn.index(os.path.basename(file))
                    if '.thumb.' not in file:
                        self.missing_thumbs.append(all_previews[file_idx])
                    item['preview'] = self.link_preview(all_previews[file_idx])
                    break
            if item.get('preview', None) is None:
                found = preview_map.get(base, None)
                if found is not None:
                    item['preview'] = self.link_preview(found)
                    debug(f'EN mapped-preview: {item["name"]}={found}')
            if item.get('preview', None) is None:
                item['preview'] = self.link_preview('html/card-no-preview.png')
                debug(f'EN missing-preview: {item["name"]}')
        self.preview_time += time.time() - t0


    def find_description(self, path, info=None):
        t0 = time.time()
        class HTMLFilter(HTMLParser):
            text = ""
            def handle_data(self, data):
                self.text += data
            def handle_endtag(self, tag):
                if tag == 'p':
                    self.text += '\n'

        if path is not None:
            fn = os.path.splitext(path)[0] + '.txt'
            if os.path.exists(fn):
                try:
                    with open(fn, "r", encoding="utf-8", errors="replace") as f:
                        txt = f.read()
                        txt = re.sub('[<>]', '', txt)
                        return txt
                except OSError:
                    pass
            if info is None:
                info = self.find_info(path)
        desc = info.get('description', '') or ''
        f = HTMLFilter()
        f.feed(desc)
        t1 = time.time()
        self.desc_time += t1-t0
        return f.text

    def find_info(self, path):
        data = {}
        if shared.cmd_opts.no_metadata:
            return data
        if path is not None:
            fn = os.path.splitext(path)[0] + '.json'
            if os.path.exists(fn):
                t0 = time.time()
                data = shared.readfile(fn, silent=True)
                if type(data) is list:
                    data = data[0]
                t1 = time.time()
                self.info_time += t1-t0
        return data


def initialize():
    shared.extra_networks.clear()


def register_page(page: ExtraNetworksPage):
    # registers extra networks page for the UI; recommend doing it in on_before_ui() callback for extensions
    debug(f'EN register-page: {page}')
    if page in shared.extra_networks:
        debug(f'EN register-page: {page} already registered')
        return
    shared.extra_networks.append(page)
    # allowed_dirs.clear()
    # for pg in shared.extra_networks:
    for folder in page.allowed_directories_for_previews():
        if folder not in allowed_dirs:
            allowed_dirs.append(os.path.abspath(folder))


def register_pages():
    debug('EN register-pages')
    from modules.ui_extra_networks_checkpoints import ExtraNetworksPageCheckpoints
    from modules.ui_extra_networks_vae import ExtraNetworksPageVAEs
    from modules.ui_extra_networks_styles import ExtraNetworksPageStyles
    register_page(ExtraNetworksPageCheckpoints())
    register_page(ExtraNetworksPageVAEs())
    register_page(ExtraNetworksPageStyles())
    if shared.opts.latent_history > 0:
        from modules.ui_extra_networks_history import ExtraNetworksPageHistory
        register_page(ExtraNetworksPageHistory())
    if shared.opts.diffusers_enable_embed:
        from modules.ui_extra_networks_textual_inversion import ExtraNetworksPageTextualInversion
        register_page(ExtraNetworksPageTextualInversion())
    if not shared.opts.lora_legacy:
        from modules.ui_extra_networks_lora import ExtraNetworksPageLora
        register_page(ExtraNetworksPageLora())
    if shared.opts.hypernetwork_enabled:
        from modules.ui_extra_networks_hypernets import ExtraNetworksPageHypernetworks
        register_page(ExtraNetworksPageHypernetworks())


def get_pages(title=None):
    visible = shared.opts.extra_networks
    pages = []
    if 'All' in visible or visible == []: # default en sort order
        visible = ['Model', 'Lora', 'Style', 'Embedding', 'VAE', 'History', 'Hypernetwork']

    titles = [page.title for page in shared.extra_networks]
    if title is None:
        for page in visible:
            try:
                idx = titles.index(page)
                pages.append(shared.extra_networks[idx])
            except ValueError:
                continue
    else:
        try:
            idx = titles.index(title)
            pages.append(shared.extra_networks[idx])
        except ValueError:
            pass
    return pages


class ExtraNetworksUi:
    def __init__(self):
        self.tabname: str = None
        self.pages: list[str] = None
        self.visible: gr.State = None
        self.state: gr.Textbox = None
        self.details: gr.Group = None
        self.details_tabs: gr.Group = None
        self.details_text: gr.Group = None
        self.tabs: gr.Tabs = None
        self.gallery: gr.Gallery = None
        self.description: gr.Textbox = None
        self.search: gr.Textbox = None
        self.button_details: gr.Button = None
        self.button_refresh: gr.Button = None
        self.button_scan: gr.Button = None
        self.button_view: gr.Button = None
        self.button_quicksave: gr.Button = None
        self.button_save: gr.Button = None
        self.button_sort: gr.Button = None
        self.button_apply: gr.Button = None
        self.button_close: gr.Button = None
        self.button_model: gr.Checkbox = None
        self.details_components: list = []
        self.last_item: dict = None
        self.last_page: ExtraNetworksPage = None
        self.state: gr.State = None


def create_ui(container, button_parent, tabname, skip_indexing = False):
    debug(f'EN create-ui: {tabname}')
    ui = ExtraNetworksUi()
    ui.tabname = tabname
    ui.pages = []
    ui.state = gr.Textbox('{}', elem_id=f"{tabname}_extra_state", visible=False)
    ui.visible = gr.State(value=False) # pylint: disable=abstract-class-instantiated
    ui.details = gr.Group(elem_id=f"{tabname}_extra_details", elem_classes=["extra-details"], visible=False)
    ui.tabs = gr.Tabs(elem_id=f"{tabname}_extra_tabs")
    ui.button_details = gr.Button('Details', elem_id=f"{tabname}_extra_details_btn", visible=False)
    state = {}

    def get_item(state, params = None):
        if params is not None and type(params) == dict:
            page = next(iter([x for x in get_pages() if x.title == 'Style']), None)
            item = page.create_style(params)
        else:
            if state is None or not hasattr(state, 'page') or not hasattr(state, 'item'):
                return None, None
            page = next(iter([x for x in get_pages() if x.title == state.page]), None)
            if page is None:
                return None, None
            item = next(iter([x for x in page.items if x["name"] == state.item]), None)
            if item is None:
                return page, None
        item = SimpleNamespace(**item)
        ui.last_item = item
        ui.last_page = page
        return page, item

    # main event that is triggered when js updates state text field with json values, used to communicate js -> python
    def state_change(state_text):
        try:
            nonlocal state
            state = SimpleNamespace(**json.loads(state_text))
        except Exception as e:
            shared.log.error(f'Networks: state error: {e}')
            return
        _page, _item = get_item(state)
        # shared.log.debug(f'Extra network: op={state.op} page={page.title if page is not None else None} item={item.filename if item is not None else None}')

    def toggle_visibility(is_visible):
        is_visible = not is_visible
        return is_visible, gr.update(visible=is_visible), gr.update(variant=("secondary-down" if is_visible else "secondary"))

    with ui.details:
        details_close = ToolButton(symbols.close, elem_id=f"{tabname}_extra_details_close", elem_classes=['extra-details-close'])
        details_close.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[ui.details])
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.HTML('<div>title</div>')
                ui.details_components.append(text)
            with gr.Column(scale=1):
                img = gr.Image(value=None, show_label=False, interactive=False, container=False, show_download_button=False, show_info=False, elem_id=f"{tabname}_extra_details_img", elem_classes=['extra-details-img'])
                ui.details_components.append(img)
                with gr.Row():
                    btn_save_img = gr.Button('Replace', elem_classes=['small-button'])
                    btn_delete_img = gr.Button('Delete', elem_classes=['small-button'])
        with gr.Group(elem_id=f"{tabname}_extra_details_tabs", visible=False) as ui.details_tabs:
            with gr.Tabs():
                with gr.Tab('Description', elem_classes=['extra-details-tabs']):
                    desc = gr.Textbox('', show_label=False, lines=8, placeholder="Network description...")
                    ui.details_components.append(desc)
                    with gr.Row():
                        btn_save_desc = gr.Button('Save', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_save_desc')
                        btn_delete_desc = gr.Button('Delete', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_delete_desc')
                        btn_close_desc = gr.Button('Close', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_close_desc')
                        btn_close_desc.click(fn=lambda: gr.update(visible=False), _js='refeshDetailsEN', inputs=[], outputs=[ui.details])
                with gr.Tab('Model metadata', elem_classes=['extra-details-tabs']):
                    info = gr.JSON({}, show_label=False)
                    ui.details_components.append(info)
                    with gr.Row():
                        btn_save_info = gr.Button('Save', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_save_info')
                        btn_delete_info = gr.Button('Delete', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_delete_info')
                        btn_close_info = gr.Button('Close', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_close_info')
                        btn_close_info.click(fn=lambda: gr.update(visible=False), _js='refeshDetailsEN', inputs=[], outputs=[ui.details])
                with gr.Tab('Embedded metadata', elem_classes=['extra-details-tabs']):
                    meta = gr.JSON({}, show_label=False)
                    ui.details_components.append(meta)
        with gr.Group(elem_id=f"{tabname}_extra_details_text", elem_classes=["extra-details-text"], visible=False) as ui.details_text:
            description = gr.Textbox(label='Description', lines=1, placeholder="Style description...")
            prompt = gr.Textbox(label='Network prompt', lines=2, placeholder="Prompt...")
            negative = gr.Textbox(label='Network negative prompt', lines=2, placeholder="Negative prompt...")
            extra = gr.Textbox(label='Network parameters', lines=2, placeholder="Generation parameters overrides...")
            wildcards = gr.Textbox(label='Wildcards', lines=2, placeholder="Wildcard prompt replacements...")
            ui.details_components += [description, prompt, negative, extra, wildcards]
            with gr.Row():
                btn_save_style = gr.Button('Save', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_save_style')
                btn_delete_style = gr.Button('Delete', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_delete_style')
                btn_close_style = gr.Button('Close', elem_classes=['small-button'], elem_id=f'{tabname}_extra_details_close_style')
                btn_close_style.click(fn=lambda: gr.update(visible=False), _js='refeshDetailsEN', inputs=[], outputs=[ui.details])

    with ui.tabs:
        def ui_tab_change(page):
            scan_visible = page in ['Model', 'Lora', 'VAE', 'Hypernetwork', 'Embedding']
            save_visible = page in ['Style']
            model_visible = page in ['Model']
            return [gr.update(visible=scan_visible), gr.update(visible=save_visible), gr.update(visible=model_visible)]

        ui.button_refresh = ToolButton(symbols.refresh, elem_id=f"{tabname}_extra_refresh")
        ui.button_scan = ToolButton(symbols.scan, elem_id=f"{tabname}_extra_scan", visible=True)
        ui.button_quicksave = ToolButton(symbols.book, elem_id=f"{tabname}_extra_quicksave", visible=False)
        ui.button_save = ToolButton(symbols.book, elem_id=f"{tabname}_extra_save", visible=False)
        ui.button_sort = ToolButton(symbols.sort, elem_id=f"{tabname}_extra_sort", visible=True)
        ui.button_view = ToolButton(symbols.view, elem_id=f"{tabname}_extra_view", visible=True)
        ui.button_close = ToolButton(symbols.close, elem_id=f"{tabname}_extra_close", visible=True)
        ui.button_model = ToolButton(symbols.refine, elem_id=f"{tabname}_extra_model", visible=True)
        ui.search = gr.Textbox('', show_label=False, elem_id=f"{tabname}_extra_search", placeholder="Search...", elem_classes="textbox", lines=2, container=False)
        ui.description = gr.Textbox('', show_label=False, elem_id=f"{tabname}_description", elem_classes=["textbox", "extra-description"], lines=2, interactive=False, container=False)

        if ui.tabname == 'txt2img': # refresh only once
            global refresh_time # pylint: disable=global-statement
            refresh_time = time.time()
        if not skip_indexing:
            import concurrent
            with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
                for page in get_pages():
                    executor.submit(page.create_items, ui.tabname)
        for page in get_pages():
            page.create_page(ui.tabname, skip_indexing)
            with gr.Tab(page.title, id=page.title.lower().replace(" ", "_"), elem_classes="extra-networks-tab") as tab:
                page_html = gr.HTML(page.patch(page.html, tabname), elem_id=f'{tabname}{page.name}_extra_page', elem_classes="extra-networks-page")
                ui.pages.append(page_html)
                tab.select(ui_tab_change, _js="getENActivePage", inputs=[ui.button_details], outputs=[ui.button_scan, ui.button_save, ui.button_model])

    def fn_save_img(image):
        if ui.last_item is None or ui.last_item.local_preview is None:
            return 'html/card-no-preview.png'
        images = []
        if ui.gallery is not None:
            images = list(ui.gallery.temp_files) # gallery cannot be used as input component so looking at most recently registered temp files
        if len(images) < 1:
            shared.log.warning(f'Extra network no image: item={ui.last_item.name}')
            return 'html/card-no-preview.png'
        try:
            images.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            image = Image.open(images[0])
        except Exception as e:
            shared.log.error(f'Extra network error opening image: item={ui.last_item.name} {e}')
            return 'html/card-no-preview.png'
        fn_delete_img(image)
        if image.width > 512 or image.height > 512:
            image = image.convert('RGB')
            image.thumbnail((512, 512), Image.Resampling.HAMMING)
        try:
            image.save(ui.last_item.local_preview, quality=50)
            shared.log.debug(f'Extra network save image: item={ui.last_item.name} filename="{ui.last_item.local_preview}"')
        except Exception as e:
            shared.log.error(f'Extra network save image: item={ui.last_item.name} filename="{ui.last_item.local_preview}" {e}')
        return image

    def fn_delete_img(_image):
        preview_extensions = ["jpg", "jpeg", "png", "webp", "tiff", "jp2"]
        fn = os.path.splitext(ui.last_item.filename)[0]
        for file in [f'{fn}{mid}{ext}' for ext in preview_extensions for mid in ['.thumb.', '.preview.', '.']]:
            if os.path.exists(file):
                os.remove(file)
                shared.log.debug(f'Extra network delete image: item={ui.last_item.name} filename="{file}"')
        return 'html/card-no-preview.png'

    def fn_save_desc(desc):
        if hasattr(ui.last_item, 'type') and ui.last_item.type == 'Style':
            params = ui.last_page.parse_desc(desc)
            if params is not None:
                fn_save_info(params)
        else:
            fn = os.path.splitext(ui.last_item.filename)[0] + '.txt'
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(desc)
            shared.log.debug(f'Extra network save desc: item={ui.last_item.name} filename="{fn}"')
        return desc

    def fn_delete_desc(desc):
        if ui.last_item is None:
            return desc
        fn = os.path.splitext(ui.last_item.filename)[0] + '.txt'
        if os.path.exists(fn):
            shared.log.debug(f'Extra network delete desc: item={ui.last_item.name} filename="{fn}"')
            os.remove(fn)
            return ''
        return desc

    def fn_save_info(info):
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        shared.writefile(info, fn, silent=True)
        shared.log.debug(f'Extra network save info: item={ui.last_item.name} filename="{fn}"')
        return info

    def fn_delete_info(info):
        if ui.last_item is None:
            return info
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        if os.path.exists(fn):
            shared.log.debug(f'Extra network delete info: item={ui.last_item.name} filename="{fn}"')
            os.remove(fn)
            return ''
        return info

    def fn_save_style(info, description, prompt, negative, extra, wildcards):
        if not isinstance(info, dict) or isinstance(info, list):
            shared.log.warning(f'Extra network save style skip: item={ui.last_item.name} not a dict: {type(info)}')
            return info
        if ui.last_item is None:
            return info
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        if hasattr(ui.last_item, 'type') and ui.last_item.type == 'Style':
            info.update(**{ 'description': description, 'prompt': prompt, 'negative': negative, 'extra': extra, 'wildcards': wildcards })
            shared.writefile(info, fn, silent=True)
            shared.log.debug(f'Extra network save style: item={ui.last_item.name} filename="{fn}"')
        return info

    def fn_delete_style(info):
        if ui.last_item is None:
            return info
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        if os.path.exists(fn):
            shared.log.debug(f'Extra network delete style: item={ui.last_item.name} filename="{fn}"')
            os.remove(fn)
            return {}
        return info

    btn_save_img.click(fn=fn_save_img, _js='closeDetailsEN', inputs=[img], outputs=[img])
    btn_delete_img.click(fn=fn_delete_img, _js='closeDetailsEN', inputs=[img], outputs=[img])
    btn_save_desc.click(fn=fn_save_desc, _js='closeDetailsEN', inputs=[desc], outputs=[desc])
    btn_delete_desc.click(fn=fn_delete_desc, _js='closeDetailsEN', inputs=[desc], outputs=[desc])
    btn_save_info.click(fn=fn_save_info, _js='closeDetailsEN', inputs=[info], outputs=[info])
    btn_delete_info.click(fn=fn_delete_info, _js='closeDetailsEN', inputs=[info], outputs=[info])
    btn_save_style.click(fn=fn_save_style, _js='closeDetailsEN', inputs=[info, description, prompt, negative, extra, wildcards], outputs=[info])
    btn_delete_style.click(fn=fn_delete_style, _js='closeDetailsEN', inputs=[info], outputs=[info])

    def show_details(text, img, desc, info, meta, description, prompt, negative, parameters, wildcards, params, _dummy1=None, _dummy2=None):
        page, item = get_item(state, params)
        valid = item is not None and hasattr(item, 'name') and hasattr(item, 'filename')
        if valid:
            stat = os.stat(item.filename) if os.path.exists(item.filename) else None
            desc = item.description
            fullinfo = shared.readfile(os.path.splitext(item.filename)[0] + '.json', silent=True)
            if 'modelVersions' in fullinfo: # sanitize massive objects
                fullinfo['modelVersions'] = []
            info = fullinfo
            meta = page.metadata.get(item.name, {}) or {}
            if type(meta) is str:
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            if ui.last_item.preview.startswith('data:'):
                b64str = ui.last_item.preview.split(',',1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(b64str)))
            elif hasattr(item, 'local_preview') and os.path.exists(item.local_preview):
                img = item.local_preview
            else:
                img = page.find_preview_file(item.filename)
            lora = ''
            model = ''
            style = ''
            note = ''
            if not os.path.exists(item.filename):
                note = f'<br>Target filename: {item.filename}'
            if page.title == 'Model':
                merge = len(list(meta.get('sd_merge_models', {})))
                if merge > 0:
                    model += f'<tr><td>Merge models</td><td>{merge} recipes</td></tr>'
                if meta.get('modelspec.architecture', None) is not None:
                    model += f'''
                        <tr><td>Architecture</td><td>{meta.get('modelspec.architecture', 'N/A')}</td></tr>
                        <tr><td>Title</td><td>{meta.get('modelspec.title', 'N/A')}</td></tr>
                        <tr><td>Resolution</td><td>{meta.get('modelspec.resolution', 'N/A')}</td></tr>
                    '''
            if page.title == 'Lora':
                try:
                    tags = getattr(item, 'tags', {})
                    tags = [f'{name}:{tags[name]}' for i, name in enumerate(tags)]
                    tags = ' '.join(tags)
                except Exception:
                    tags = ''
                try:
                    triggers = ' '.join(info.get('tags', []))
                except Exception:
                    triggers = ''
                lora = f'''
                    <tr><td>Model tags</td><td>{tags}</td></tr>
                    <tr><td>User tags</td><td>{triggers}</td></tr>
                    <tr><td>Base model</td><td>{meta.get('ss_sd_model_name', 'N/A')}</td></tr>
                    <tr><td>Resolution</td><td>{meta.get('ss_resolution', 'N/A')}</td></tr>
                    <tr><td>Training images</td><td>{meta.get('ss_num_train_images', 'N/A')}</td></tr>
                    <tr><td>Comment</td><td>{meta.get('ss_training_comment', 'N/A')}</td></tr>
                '''
            if page.title == 'Style':
                description = item.description
                prompt = item.prompt
                negative = item.negative
                parameters = item.extra
                wildcards = item.wildcards
                style = f'''
                    <tr><td>Name</td><td>{item.name}</td></tr>
                    <tr><td>Description</td><td>{item.description}</td></tr>
                    <tr><td>Preview Embedded</td><td>{item.preview.startswith('data:')}</td></tr>
                '''
                # desc = f'Name: {os.path.basename(item.name)}\nDescription: {item.description}\nPrompt: {item.prompt}\nNegative: {item.negative}\nExtra: {item.extra}\n'
            if item.name.startswith('Diffusers'):
                url = item.name.replace('Diffusers/', '')
                url = f'<a href="https://huggingface.co/{url}" target="_blank">https://huggingface.co/models/{url}</a>' if url is not None else 'N/A'
            else:
                url = info.get('id', None) if info is not None else None
                url = f'<a href="https://civitai.com/models/{url}" target="_blank">civitai.com/models/{url}</a>' if url is not None else 'N/A'
            text = f'''
                <h2 style="border-bottom: 1px solid var(--button-primary-border-color); margin: 0em 0px 1em 0 !important">{item.name}</h2>
                <table style="width: 100%; line-height: 1.5em;"><tbody>
                    <tr><td>Type</td><td>{page.title}</td></tr>
                    <tr><td>Alias</td><td>{getattr(item, 'alias', 'N/A')}</td></tr>
                    <tr><td>Filename</td><td>{item.filename}</td></tr>
                    <tr><td>Hash</td><td>{getattr(item, 'hash', 'N/A')}</td></tr>
                    <tr><td>Size</td><td>{round(stat.st_size/1024/1024, 2) if stat is not None else 'N/A'} MB</td></tr>
                    <tr><td>Last modified</td><td>{datetime.fromtimestamp(stat.st_mtime) if stat is not None else 'N/A'}</td></tr>
                    <tr><td>Source URL</td><td>{url}</td></tr>
                    <tr><td style="border-top: 1px solid var(--button-primary-border-color);"></td><td></td></tr>
                    {lora}
                    {model}
                    {style}
                </tbody></table>
                {note}
            '''
        return [
            text, # gr.html
            img, # gr.image
            desc, # gr.textbox
            info, # gr.json
            meta, # gr.json
            description, # gr.textbox
            prompt, # gr.textbox
            negative, # gr.textbox
            parameters, # gr.textbox
            wildcards, # gr.textbox
            gr.update(visible=valid), # details ui visible
            gr.update(visible=page is not None and page.title != 'Style'), # details ui tabs visible
            gr.update(visible=page is not None and page.title == 'Style'), # details ui text visible
        ]

    def ui_refresh_click(title):
        pages = []
        for page in get_pages():
            if page.title != title:
                pages.append(page.html)
                continue
            page.page_time = 0
            page.refresh_time = 0
            page.refresh()
            page.create_page(ui.tabname)
            shared.log.debug(f"Networks: refresh page='{page.title}' items={len(page.items)} tab={ui.tabname}")
            pages.append(page.html)
        ui.search.update(title)
        return pages

    def ui_view_cards(title):
        pages = []
        for page in get_pages():
            shared.opts.extra_networks_view = page.view
            # shared.opts.save(shared.config_filename)
            page.view = 'gallery' if page.view == 'list' else 'list'
            page.card = card_full if page.view == 'gallery' else card_list
            page.html = ''
            page.create_page(ui.tabname)
            shared.log.debug(f"Networks: refresh page='{page.title}' items={len(page.items)} tab={ui.tabname} view={page.view}")
            pages.append(page.html)
        ui.search.update(title)
        return pages

    def ui_scan_click(title):
        from modules import ui_models
        if ui_models.search_metadata_civit is not None:
            ui_models.search_metadata_civit(True, title)
        return ui_refresh_click(title)

    def ui_save_click():
        filename = os.path.join(paths.data_path, "params.txt")
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf8") as file:
                prompt = file.read()
        else:
            prompt = ''
        params = infotext.parse(prompt)
        res = show_details(text=None, img=None, desc=None, info=None, meta=None, parameters=None, description=None, prompt=None, negative=None, wildcards=None, params=params)
        return res

    def ui_quicksave_click(name):
        if name is None or len(name) < 1:
            shared.log.warning("Network quick save style: no name provided")
            return
        fn = os.path.join(paths.data_path, "params.txt")
        if os.path.exists(fn):
            with open(fn, "r", encoding="utf8") as file:
                prompt = file.read()
        else:
            prompt = ''
        params = infotext.parse(prompt)
        fn = os.path.join(shared.opts.styles_dir, os.path.splitext(name)[0] + '.json')
        prompt = params.get('Prompt', '')
        item = {
            "name": name,
            "description": '',
            "prompt": prompt,
            "negative": params.get('Negative prompt', ''),
            "extra": '',
        }
        shared.writefile(item, fn, silent=True)
        if len(prompt) > 0:
            shared.log.debug(f"Network quick save style: item={name} filename='{fn}'")
        else:
            shared.log.warning(f"Network quick save model: item={name} filename='{fn}' prompt is empty")

    def ui_sort_cards(sort_order):
        if shared.opts.extra_networks_sort != sort_order:
            shared.opts.extra_networks_sort = sort_order
            shared.opts.save(shared.config_filename)
        return f'Networks: sort={sort_order}'

    dummy = gr.State(value=False) # pylint: disable=abstract-class-instantiated
    button_parent.click(fn=toggle_visibility, inputs=[ui.visible], outputs=[ui.visible, container, button_parent])
    ui.button_close.click(fn=toggle_visibility, inputs=[ui.visible], outputs=[ui.visible, container])
    ui.button_sort.click(fn=ui_sort_cards, _js='sortExtraNetworks', inputs=[ui.search], outputs=[ui.description])
    ui.button_view.click(fn=ui_view_cards, inputs=[ui.search], outputs=ui.pages)
    ui.button_refresh.click(fn=ui_refresh_click, _js='getENActivePage', inputs=[ui.search], outputs=ui.pages)
    ui.button_scan.click(fn=ui_scan_click, _js='getENActivePage', inputs=[ui.search], outputs=ui.pages)
    ui.button_save.click(fn=ui_save_click, inputs=[], outputs=ui.details_components + [ui.details])
    ui.button_quicksave.click(fn=ui_quicksave_click, _js="() => prompt('Prompt name', '')", inputs=[ui.search], outputs=[])
    ui.button_details.click(show_details, _js="getCardDetails", inputs=ui.details_components + [dummy, dummy, dummy], outputs=ui.details_components + [ui.details, ui.details_tabs, ui.details_text])
    ui.state.change(state_change, inputs=[ui.state], outputs=[])
    return ui


def setup_ui(ui, gallery):
    ui.gallery = gallery
