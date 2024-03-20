import os
from typing import List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocket, WebSocketState, WebSocketDisconnect
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from modules import shared, files_cache


debug = shared.log.debug if os.environ.get('SD_BROWSER_DEBUG', None) is not None else lambda *args, **kwargs: None


OPTS_FOLDERS = [
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_control_samples",
    "outdir_extras_samples",
    "outdir_save"
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

    @app.get('/sdapi/v1/browser/folders', response_model=List[str])
    def get_folders():
        folders = [shared.opts.data.get(f, '') for f in OPTS_FOLDERS]
        folders += list(shared.opts.browser_folders.split(','))
        folders = [f.strip() for f in folders if f != '']
        folders = list(dict.fromkeys(folders)) # filter duplicates
        folders = [f for f in folders if os.path.isdir(f)]
        if shared.demo is not None:
            for f in folders:
                if os.path.isabs(f) and f not in shared.demo.allowed_paths:
                    debug(f'Browser folders allow: {f}')
                    shared.demo.allowed_paths.append(f)
        debug(f'Browser folders: {folders}')
        return JSONResponse(content=folders)

    @app.websocket("/sdapi/v1/browser/files")
    async def ws_files(ws: WebSocket):
        try:
            await manager.connect(ws)
            folder = await ws.receive_text()
            debug(f'Browser WS folder: {folder}')
            for f in files_cache.directory_files(folder, recursive=True):
                file = os.path.relpath(f, folder)
                stat = os.stat(f)
                dct = {
                    'folder': folder,
                    'file': file,
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                }
                await manager.send(ws, dct)
            await manager.send(ws, '#END#')
        except WebSocketDisconnect:
            debug('Browser WS unexpected disconnect')
        manager.disconnect(ws)

    @app.websocket("/sdapi/v1/browser/file/{file}")
    async def ws_file(ws: WebSocket, file: str):
        try:
            await manager.connect(ws)
            with open(file, 'rb') as f:  # noqa: ASYNC101
                await manager.send(ws, f.read())
        except WebSocketDisconnect:
            manager.disconnect(ws)
