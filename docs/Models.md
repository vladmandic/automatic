# Models

List of popular text-to-image generative models with their respective parameters and architecture overview  
Original URL: <https://github.com/vladmandic/automatic/wiki/Models>

| Publisher | Model | Version | Size | Diffusion Architecture | Model Params | Text Encoder(s) | TE Params | Auto Encoder | Other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| StabilityAI | Stable Diffusion | 1.5 | 2.28GB | UNet | 0.86B | CLiP ViT-L | 0.12B | VAE | |
| StabilityAI | Stable Diffusion | 2.1 | 2.58GB | UNet | 0.86B | CLiP ViT-H | 0.34B | VAE | |
| StabilityAI | Stable Diffusion | XL | 6.94GB | UNet | 2.56B | CLiP ViT-L + ViT+G | 0.12B + 0.69B | VAE | |
| StabilityAI | Stable Diffusion | 3.0 Medium | 15.14GB | MMDiT | 2.0B | CLiP ViT-L + ViT+G + T5-XXL | 0.12B + 0.69B + 4.76B | 16ch VAE | |
| StabilityAI | Stable Diffusion | 3.5 Medium | 15.89GB | MMDiT | 2.25B | CLiP ViT-L + ViT+G + T5-XXL | 0.12B + 0.69B + 4.76B | 16ch VAE | |
| StabilityAI | Stable Diffusion | 3.5 Large | 26.98GB | MMDiT | 8.05B | CLiP ViT-L + ViT+G + T5-XXL | 0.12B + 0.69B + 4.76B | 16ch VAE | |
| StabilityAI | Stable Cascade | Medium | 11.82GB | Multi-stage UNet | 1.56B + 3.6B | CLiP ViT-G | 0.69B | 42x VQE | |
| StabilityAI | Stable Cascade | Lite | 4.97GB | Multi-stage UNet | 0.7B + 1.0B | CLiP ViT-G | 0.69B | 42x VQE | |
| Black Forest Labs | Flux | 1 Dev/Schnell| 32.93GB | MMDiT | 11.9B | CLiP ViT-L + T5-XXL | 0.12B + 4.769B | 16ch VAE | |
| FAL | AuraFlow | 0.3 | 31.90GB | MMDiT | 6.8B | UMT5 | 12.1B | VAE | |
| AlphaVLLM | Lumina | Next SFT | 8.67GB | DiT | 1.7B | Gemma | 2.5B | VAE | LM |
| PixArt | Alpha | XL 2 | 21.3GB | DiT | 0.61B | T5-XXL | 4.76B | VAE | |
| PixArt | Sigma | XL 2 | 21.3GB | DiT | 0.61B | T5-XXL | 4.76B | VAE | |
| Segmind | SSD-1B | N/A | 8.72GB | UNet | 1.33B | CLiP ViT-L + ViT+G | 0.12B + 0.69B | VAE | |
| Segmind | Vega | N/A | 6.43GB | UNet | 0.75B | CLiP ViT-L + ViT+G | 0.12B + 0.69B | VAE | |
| Segmind | Tiny | N/A | 1.03GB | UNet | 0.32B | CLiP ViT-L | 0.12B | VAE | |
| Kwai | Kolors | N/A | 17.40GB | UNnet | 2.58B | ChatGLM | 6.24B | VAE | LM |
| PlaygroundAI | Playground | 1.0 | 4.95GB| UNet | 0.86B | CLiP ViT-L | 0.12B | VAE | |
| PlaygroundAI | Playground | 2.x | 13.35GB | UNet | 2.56B | CLiP ViT-L + ViT+G | 0.12B + 0.69B | VAE | |
| Tencent | HunyuanDiT | 1.2 | 14.09GB | DiT | 1.5B | BERT + T5-XL | 3.52B + 1.67B | VAE | LM |
| Warp AI | Wuerstchen | N/A | 12.16GB | Multi-stage UNet | 1.0B + 1.05B | CLiP ViT-L + ViT+G | 0.12B + 0.69B | 42x VQE | |
| Kandinsky | Kandinsky | 2.2 | 5.15GB | Unet | 1.25B | CLiP ViT-G | 0.69B | VQ | |
| Kandinsky | Kandinsky | 3.0 | 27.72GB | Unet | 3.05B | T5-XXXL | 8.72B | VQ | |
| Thudm | CogView | 3 Plus | 24.96GB | DiT | 2.85B | T5-XXL | 4.76B | VAE | |
| IDKiro | SDXS | N/A | 2.05GB | UNet | 0.32B | CLiP ViT-L | 0.12B | VAE | |
| Open-MUSE | aMUSEd | 256 | 3.41GB | ViT | 0.60B | CLiP ViT-L | 0.12B | VQ | |
| Koala | Koala | 700M | 6.58GB | UNet | 0.78B | CLiP ViT-L + ViT+G | 0.12B + 0.69B | VAE | |
| Thu-ML | UniDiffuser | v1 | 5.37GB | U-ViT | 0.95B | CLiP ViT-L + CLiP ViT-B | 0.12B + 0.16B | VAE | |
| Salesforce | BLIP-Diffusion | N/A | 7.23GB | UNet | 0.86B | CLiP ViT-L + BLiP-2 | 0.12B + 0.49B | VAE | |
| DeepFloyd | IF | M | 12.79GB | Multi-stage UNet | 0.37B + 0.46B | T5-XXL | 4.76B | Pixel | |
| DeepFloyd | IF | L | 15.48GB | Multi-stage UNet | 0.61B + 0.93B | T5-XXL | 4.76B | Pixel | |
| MeissonFlow | Meissonic | N/A | 3.64GB | DiT | 1.18B | CLiP ViT-H | 0.35B | VQ | |
| VectorSpaceLab | OmniGen | v1 | 15.47GB | Transformer | 3.76B | None | 0 | VAE | Phi-3 |

## Notes

- Created using [SD.Next](https://github.com/vladmandic/automatic/) built-in model analyzer  
- Number of parameters is proportional to model complexity and ability to learn  
  Quality of generated images is also influenced by training data and duration of training  
- Size refers to original model variant in 16bit precision where available  
  Quantized variations may be smaller  
- Distilled variants are not included as typical goal-distilling does not change underlying model params  
  e.g. Turbo/LCM/Hyper/Lightning/etc. or even Dev/Schnell  
