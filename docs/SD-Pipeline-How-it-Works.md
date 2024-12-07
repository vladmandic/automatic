# Stable Diffusion Pipeline

This is probably the best end-to-end semi-technical article:  
<https://stable-diffusion-art.com/how-stable-diffusion-work/>

And a detailed look at diffusion process:
<https://towardsdatascience.com/understanding-diffusion-probabilistic-models-dpms-1940329d6048>

But this is a short look at the pipeline:

1. Encoder / Conditioning
   Text (via tokenizer) or image (via vision model) to semantic map  
   (e.g CLiP text encoder)  
2. Sampler
   Generate noise which is starting point to map to content  
   (e.g. k_lms)  
3. Diffuser
   Create vector content based on resolved noise + semantic map  
   (e.g. actual stable diffusion checkpoint)  
4. Autoencoder
   Maps between latent and pixel space (actually creates images from vectors)  
   (e.g. typically some image-database trained GAN)  
5. Denoising
   Get meaningful images from pixel signatures  
   Basically, blends what autoencoder inserted using information from diffuser  
   (e.g. U-NET)
6. Loop and repeat
   From step#3 with cross-attention to blend results  
7. Run additional models as needed  
   - Upscale (e.g. ESRGAN)  
   - Resore Face (e.g. GFPGAN or CodeFormer)  
