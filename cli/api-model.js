#!/usr/bin/env node

const sd_url = process.env.SDAPI_URL || 'http://127.0.0.1:7860';
const sd_username = process.env.SDAPI_USR;
const sd_password = process.env.SDAPI_PWD;
const models = [
  '/mnt/models/stable-diffusion/sd15/lyriel_v16.safetensors',
  '/mnt/models/stable-diffusion/flux/flux-finesse_v2-f1h-fp8.safetensors',
  '/mnt/models/stable-diffusion/sdxl/TempestV0.1-Artistic.safetensors',
];

async function options(data) {
  const method = 'POST';
  const headers = new Headers();
  const body = JSON.stringify(data);
  headers.set('Content-Type', 'application/json');
  if (sd_username && sd_password) headers.set({ Authorization: `Basic ${btoa('sd_username:sd_password')}` });
  const res = await fetch(`${sd_url}/sdapi/v1/options`, { method, headers, body });
  return res;
}

async function main() {
  for (const model of models) {
    console.log('model:', model);
    const res = await options({ sd_model_checkpoint: model });
    console.log('result:', res);
  }
}

main();
