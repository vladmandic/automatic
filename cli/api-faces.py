#!/usr/bin/env python
import os
import io
import base64
import logging
import argparse
import requests
import urllib3
from PIL import Image

sd_url = os.environ.get('SDAPI_URL', "http://127.0.0.1:7860")
sd_username = os.environ.get('SDAPI_USR', None)
sd_password = os.environ.get('SDAPI_PWD', None)

logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def auth():
    if sd_username is not None and sd_password is not None:
        return requests.auth.HTTPBasicAuth(sd_username, sd_password)
    return None


def post(endpoint: str, dct: dict = None):
    req = requests.post(f'{sd_url}{endpoint}', json = dct, timeout=300, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


def encode(f):
    image = Image.open(f)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    with io.BytesIO() as stream:
        image.save(stream, 'JPEG')
        image.close()
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def detect(args): # pylint: disable=redefined-outer-name
    data = post('/sdapi/v1/faces', { 'image': encode(args.image) })
    for face in zip(data['images'], data['scores']):
        log.info(f'Face: score={face[1]}')
        image = Image.open(io.BytesIO(base64.b64decode(face[0])))
        image.save(f'/tmp/face_{face[1]}.jpg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'api-faces')
    parser.add_argument('--image', required=True, help='input image')
    args = parser.parse_args()
    log.info(f'api-faces: {args}')
    detect(args)
