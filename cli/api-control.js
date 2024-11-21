#!/usr/bin/env node

// simple nodejs script to test sdnext api

const fs = require('fs');
const process = require('process');

const sd_url = process.env.SDAPI_URL || 'http://127.0.0.1:7860';
const sd_username = process.env.SDAPI_USR;
const sd_password = process.env.SDAPI_PWD;
const sd_options = {
  // first pass
  prompt: 'beautiful lady, in the steampunk style',
  negative_prompt: 'foggy, blurry',
  seed: -1,
  steps: 20,
  batch_size: 1,
  n_iter: 1,
  cfg_scale: 6,
  width: 1280,
  height: 800,
  // api return options
  save_images: false,
  send_images: true,
  script_name: 'pulid',
};

async function main() {
  const method = 'POST';
  const headers = new Headers();
  const body = JSON.stringify(sd_options);
  headers.set('Content-Type', 'application/json');
  if (sd_username && sd_password) headers.set({ Authorization: `Basic ${btoa('sd_username:sd_password')}` });
  const res = await fetch(`${sd_url}/sdapi/v1/txt2img`, { method, headers, body });

  if (res.status !== 200) {
    console.log('Error', res.status);
  } else {
    const json = await res.json();
    console.log('result:', json.info);
    for (const i in json.images) { // eslint-disable-line guard-for-in
      const file = `/tmp/test-${i}.jpg`;
      const data = atob(json.images[i])
      fs.writeFileSync(file, data, 'binary');
      console.log('image saved:', file);
    }
  }
}

main();
