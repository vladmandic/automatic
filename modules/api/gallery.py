import io
import os
import time
import base64
from typing import List
from urllib.parse import quote, unquote
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocket, WebSocketState, WebSocketDisconnect
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from PIL import Image
from modules import shared, images, files_cache


debug = shared.log.debug if os.environ.get('SD_BROWSER_DEBUG', None) is not None else lambda *args, **kwargs: None


OPTS_FOLDERS = [
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_control_samples",
    "outdir_extras_samples",
    "outdir_save",
    "outdir_video",
    "outdir_init_images",
    "outdir_grids",
    "outdir_txt2img_grids",
    "outdir_img2img_grids",
    "outdir_control_grids",
]

### class definitions

class ReqFiles(BaseModel):
    folder: str = Field(title="Folder")

### ws connection manager

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        agent = ws._headers.get("user-agent", "") # pylint: disable=protected-access
        debug(f'Browser WS connect: client={ws.client.host} agent="{agent}"')
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        debug(f'Browser WS disconnect: client={ws.client.host}')
        self.active.remove(ws)

    async def send(self, ws: WebSocket, data: str|dict|bytes):
        # debug(f'Browser WS send: client={ws.client.host} data={type(data)}')
        if ws.client_state != WebSocketState.CONNECTED:
            return
        if isinstance(data, bytes):
            await ws.send_bytes(data)
        elif isinstance(data, dict):
            await ws.send_json(data)
        elif isinstance(data, str):
            await ws.send_text(data)
        else:
            debug(f'Browser WS send: client={ws.client.host} data={type(data)} unknown')

    async def broadcast(self, data: str|dict|bytes):
        for ws in self.active:
            await self.send(ws, data)

### api definitions

def register_api(app: FastAPI): # register api
    manager = ConnectionManager()

    def get_video_thumbnail(filepath):
        from modules.ui_control_helpers import get_video_params
        try:
            frames, fps, duration, width, height, codec, frame = get_video_params(filepath, capture=True)
            h = shared.opts.extra_networks_card_size
            w = shared.opts.extra_networks_card_size if shared.opts.browser_fixed_width else width * h // height
            frame = frame.convert('RGB')
            frame.thumbnail((w, h), Image.Resampling.HAMMING)
            buffered = io.BytesIO()
            frame.save(buffered, format='jpeg')
            data_url = f'data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode("ascii")}'
            frame.close()
            content = {
                'exif': f'Codec: {codec}, Frames: {frames}, Duration: {duration:.2f} sec, FPS: {fps:.2f}',
                'data': data_url,
                'width': width,
                'height': height,
            }
            return content
        except Exception as e:
            shared.log.error(f'Gallery video: file="{filepath}" {e}')
            return {}

    @app.get('/sdapi/v1/browser/folders', response_model=List[str])
    def get_folders():
        folders = [shared.opts.data.get(f, '') for f in OPTS_FOLDERS]
        folders += list(shared.opts.browser_folders.split(','))
        folders = [f.strip() for f in folders if f != '']
        folders = list(dict.fromkeys(folders)) # filter duplicates
        folders = [f for f in folders if os.path.isdir(f)]
        if shared.demo is not None:
            for f in folders:
                if f not in shared.demo.allowed_paths:
                    debug(f'Browser folders allow: {f}')
                    shared.demo.allowed_paths.append(quote(f))
        debug(f'Browser folders: {folders}')
        return JSONResponse(content=folders)

    @app.get("/sdapi/v1/browser/thumb", response_model=dict)
    async def get_thumb(file: str):
        try:
            decoded = unquote(file)
            if decoded.lower().endswith('.mp4'):
                content = get_video_thumbnail(decoded)
                return JSONResponse(content=content)
            else:
                image = Image.open(decoded)
                geninfo, _items = images.read_info_from_image(image)
                h = shared.opts.extra_networks_card_size
                w = shared.opts.extra_networks_card_size if shared.opts.browser_fixed_width else image.width * h // image.height
                width, height = image.width, image.height
                image = image.convert('RGB')
                image.thumbnail((w, h), Image.Resampling.HAMMING)
                buffered = io.BytesIO()
                image.save(buffered, format='jpeg')
                data_url = f'data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode("ascii")}'
                image.close()
                content = {
                    'exif': geninfo,
                    'data': data_url,
                    'width': width,
                    'height': height,
                }
                return JSONResponse(content=content)
        except Exception as e:
            shared.log.error(f'Gallery: {file} {e}')
            content = { 'error': str(e) }
            return JSONResponse(content=content)

    @app.websocket("/sdapi/v1/browser/files")
    async def ws_files(ws: WebSocket):
        try:
            await manager.connect(ws)
            folder = await ws.receive_text()
            t0 = time.time()
            numFiles = 0
            for f in files_cache.directory_files(folder, recursive=True):
                numFiles += 1
                file = os.path.relpath(f, folder)
                stat = os.stat(f)
                dct = {
                    'folder': quote(folder),
                    'file': quote(file),
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                }
                await manager.send(ws, dct)
            await manager.send(ws, '#END#')
            t1 = time.time()
            shared.log.debug(f'Gallery: folder="{folder}" files={numFiles} time={t1-t0:.3f}')
        except WebSocketDisconnect:
            debug('Browser WS unexpected disconnect')
        manager.disconnect(ws)
