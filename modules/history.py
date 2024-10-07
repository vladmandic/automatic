import sys
import datetime
from collections import deque
from modules import shared, devices


class Item():
    def __init__(self, latent, preview=None, meta=None):
        self.ts = datetime.datetime.now()
        self.latent = latent.detach().clone().to(devices.cpu)
        self.preview = preview
        self.meta = meta


class History():
    def __init__(self):
        self.latents = deque(maxlen=1000)
        shared.log.debug(f'History init: max={shared.opts.latent_history}')

    @property
    def count(self):
        return len(self.latents)

    @property
    def size(self):
        s = 0
        for item in self.latents:
            s += sys.getsizeof(item.latent.storage())
        return s

    @property
    def list(self):
        return [item.ts for item in self.latents]

    @property
    def latest(self):
        return self.get(0)

    def add(self, latent, preview=None, meta=None):
        item = Item(latent, preview, meta)
        self.latents.appendleft(item)
        shared.log.debug(f'History add: shape={latent.shape} dtype={latent.dtype} count={self.count}')
        if self.count >= shared.opts.latent_history:
            self.latents.pop()

    def get(self, index: int = 0):
        item = self.latents[index]
        shared.log.debug(f'History get: index={index} time={item.ts} shape={item.latent.shape} dtype={item.latent.dtype} count={self.count}')
        return item.latent.to(devices.device)

    def clear(self):
        self.latents.clear()
        shared.log.debug(f'History clear: count={self.count}')

    def load(self):
        pass

    def save(self):
        pass
