#!/usr/bin/env python
import io
import os
import sys
import base64
from PIL import Image
from rich import print # pylint: disable=redefined-builtin


def encode(file: str):
    image = Image.open(file) if os.path.exists(file) else None
    print(f'Input: file={file} image={image}')
    if image is None:
        return None
    if image.mode != 'RGB':
        image = image.convert('RGB')
    with io.BytesIO() as stream:
        image.save(stream, 'JPEG')
        image.close()
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


if __name__ == "__main__":
    sys.argv.pop(0)
    fn = sys.argv[0] if len(sys.argv) > 0 else ''
    b64 = encode(fn)
    print('=== BEGIN ===')
    print(f'{b64}')
    print('=== END ===')

