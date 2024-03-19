#!/bin/env python

import os
import io
import re
import sys
import json
from PIL import Image, ExifTags, TiffImagePlugin, PngImagePlugin
from rich import print # pylint: disable=redefined-builtin


def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def parse_generation_parameters(infotext): # copied from modules.generation_parameters_copypaste
    if not isinstance(infotext, str):
        return {}

    re_param = re.compile(r'\s*([\w ]+):\s*("(?:\\"[^,]|\\"|\\|[^\"])+"|[^,]*)(?:,|$)') # multi-word: value
    re_size = re.compile(r"^(\d+)x(\d+)$") # int x int
    sanitized = infotext.replace('prompt:', 'Prompt:').replace('negative prompt:', 'Negative prompt:').replace('Negative Prompt', 'Negative prompt') # cleanup everything in brackets so re_params can work
    sanitized = re.sub(r'<[^>]*>', lambda match: ' ' * len(match.group()), sanitized)
    sanitized = re.sub(r'\([^)]*\)', lambda match: ' ' * len(match.group()), sanitized)
    sanitized = re.sub(r'\{[^}]*\}', lambda match: ' ' * len(match.group()), sanitized)

    params = dict(re_param.findall(sanitized))
    params = { k.strip():params[k].strip() for k in params if k.lower() not in ['hashes', 'lora', 'embeddings', 'prompt', 'negative prompt']} # remove some keys
    first_param = next(iter(params)) if params else None
    params_idx = sanitized.find(f'{first_param}:') if first_param else -1
    negative_idx = infotext.find("Negative prompt:")

    prompt = infotext[:params_idx] if negative_idx == -1 else infotext[:negative_idx] # prompt can be with or without negative prompt
    negative = infotext[negative_idx:params_idx] if negative_idx >= 0 else ''

    for k, v in params.copy().items(): # avoid dict-has-changed
        if len(v) > 0 and v[0] == '"' and v[-1] == '"':
            v = unquote(v)
        m = re_size.match(v)
        if v.replace('.', '', 1).isdigit():
            params[k] = float(v) if '.' in v else int(v)
        elif v == "True":
            params[k] = True
        elif v == "False":
            params[k] = False
        elif m is not None:
            params[f"{k}-1"] = int(m.group(1))
            params[f"{k}-2"] = int(m.group(2))
        elif k == 'VAE' and v == 'TAESD':
            params["Full quality"] = False
        else:
            params[k] = v
    params["Prompt"] = prompt.replace('Prompt:', '').strip()
    params["Negative prompt"] = negative.replace('Negative prompt:', '').strip()
    return params


class Exif: # pylint: disable=single-string-used-for-slots
    __slots__ = ('__dict__') # pylint: disable=superfluous-parens
    def __init__(self, image = None):
        super(Exif, self).__setattr__('exif', Image.Exif()) # pylint: disable=super-with-arguments
        self.pnginfo = PngImagePlugin.PngInfo()
        self.tags = {**dict(ExifTags.TAGS.items()), **dict(ExifTags.GPSTAGS.items())}
        self.ids = {**{v: k for k, v in ExifTags.TAGS.items()}, **{v: k for k, v in ExifTags.GPSTAGS.items()}}
        if image is not None:
            self.load(image)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        return self.exif.get(attr, None)

    def load(self, img: Image):
        img.load() # exif may not be ready
        exif_dict = {}
        try:
            exif_dict = dict(img._getexif().items()) # pylint: disable=protected-access
        except Exception:
            exif_dict = dict(img.info.items())
        for key, val in exif_dict.items():
            if isinstance(val, bytes): # decode bytestring
                val = self.decode(val)
            if val is not None:
                if isinstance(key, str):
                    self.exif[key] = val
                    self.pnginfo.add_text(key, str(val), zip=False)
                elif isinstance(key, int) and key in ExifTags.TAGS: # add known tags
                    if self.tags[key] in ['ExifOffset']:
                        continue
                    self.exif[self.tags[key]] = val
                    self.pnginfo.add_text(self.tags[key], str(val), zip=False)
                    # if self.tags[key] == 'UserComment': # add geninfo from UserComment
                        # self.geninfo = val
                else:
                    print('metadata unknown tag:', key, val)
        for key, val in self.exif.items():
            if isinstance(val, bytes): # decode bytestring
                self.exif[key] = self.decode(val)

    def decode(self, s: bytes):
        remove_prefix = lambda text, prefix: text[len(prefix):] if text.startswith(prefix) else text # pylint: disable=unnecessary-lambda-assignment
        for encoding in ['utf-8', 'utf-16', 'ascii', 'latin_1', 'cp1252', 'cp437']: # try different encodings
            try:
                s = remove_prefix(s, b'UNICODE')
                s = remove_prefix(s, b'ASCII')
                s = remove_prefix(s, b'\x00')
                val = s.decode(encoding, errors="strict")
                val = re.sub(r'[\x00-\x09]', '', val).strip() # remove remaining special characters
                if len(val) == 0: # remove empty strings
                    val = None
                return val
            except Exception:
                pass
        return None

    def parse(self):
        x = self.exif.pop('parameters', None) or self.exif.pop('UserComment', None)
        res = parse_generation_parameters(x)
        return res

    def get_bytes(self):
        ifd = TiffImagePlugin.ImageFileDirectory_v2()
        exif_stream = io.BytesIO()
        for key, val in self.exif.items():
            if key in self.ids:
                ifd[self.ids[key]] = val
            else:
                print('metadata unknown exif tag:', key, val)
        ifd.save(exif_stream)
        raw = b'Exif\x00\x00' + exif_stream.getvalue()
        return raw


def read_exif(filename: str):
    if filename.lower().endswith('.heic'):
        from pi_heif import register_heif_opener
        register_heif_opener()
    try:
        image = Image.open(filename)
        exif = Exif(image)
        print('image:', filename, 'format:', image)
        print('exif:', vars(exif.exif)['_data'])
        print('info:', exif.parse())
    except Exception as e:
        print('metadata error reading:', filename, e)


if __name__ == '__main__':
    sys.argv.pop(0)
    if len(sys.argv) == 0:
        print('metadata:', 'no files specified')
    for fn in sys.argv:
        if os.path.isfile(fn):
            read_exif(fn)
        elif os.path.isdir(fn):
            for root, _dirs, files in os.walk(fn):
                for file in files:
                    read_exif(os.path.join(root, file))
