# Offload

Offload is a method of moving model or parts of the model between the GPU memory (VRAM) and system memory (RAM) in order to reduce the memory footprint of the model and allow it to run on GPUs with lower VRAM.

## Automatic offload

!!! tip

    Automatic offload is set by the *Settings -> Diffusers -> Model offload mode*  

### Balanced

Balanced offload works differently than all other offloading methods as it performs offloading only when the VRAM usage exceeds the user-specified threshold.

- Recommended for compatible high VRAM GPUs  
- Faster but requires compatible platform and sufficient VRAM  
- Balanced offload moves parts of the model depending on the user-specified threshold  
  allowing to control how much VRAM is to be used  
- Default memory threshold is 75% of the available GPU memory  
  Configure threshold in *Settings -> Diffusers -> Max GPU memory for balanced offload mode in GB*

!!! warning

    Not compatible with *Optimum.Quanto* `qint` quantization  

### Sequential

Works on layer-by-layer basis of each model component that is marked as offload-compatible  

- Recommended for low VRAM GPUs
- Much slower but allows to run large models such as FLUX even on GPUs with 6GB VRAM  

!!! warning

    Not compatible with Quanto `qint` or BitsAndBytes `nf4` quantization  

!!! note

    Use of `--lowvram` automatically triggers use of sequenential offload

## Model

Works on model component level by offloading components that are marked as offload-compatible  
For example, VAE, text-encoder, etc.

- Recommended for medium when balanced offload is not compatible  
- Higher compatibility than either balanced and sequential, but lesser savings  

Limitations: N/A

!!! note

    Use of `--medvram` automatically triggers use of model offload

## Manual Offload

In addition to above mentioned automatic offload method, SD.Next includes manual offload methods which are less granular and are only supported for specific models.

- Move base model to CPU when using refiner
- Move base model to CPU when using VAE
- Move refiner model to CPU when not in use
