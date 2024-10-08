#!/usr/bin/env python
import os
import time
import base64
import logging
import argparse
import requests
import urllib3


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


def get(endpoint: str, dct: dict = None):
    req = requests.get(f'{sd_url}{endpoint}', json=dct, timeout=300, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


def post(endpoint: str, dct: dict = None):
    req = requests.post(f'{sd_url}{endpoint}', json = dct, timeout=300, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


def info(args): # pylint: disable=redefined-outer-name
    t0 = time.time()
    with open(args.input, 'rb') as f:
        content = f.read()
    dct = { 'image': base64.b64encode(content).decode() }
    if args.model is not None:
        dct['model'] = args.model
    if args.question is not None:
        dct['question'] = args.question
    data = post('/sdapi/v1/vqa', dct)
    t1 = time.time()
    log.info(f'answer: {data} time={t1-t0:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'api-vqa')
    parser.add_argument('--input', required=True, help='input image')
    parser.add_argument('--model', required=False, help='vqa model')
    parser.add_argument('--question', required=False, help='question')
    args = parser.parse_args()
    log.info(f'api-vqa: {args}')
    info(args)
