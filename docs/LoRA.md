# LoRA

## What is LoRA?

LoRA, short for Low-Rank Adaptation, is a method used in Generative AI models to fine-tune the model with specific styles or concepts while keeping the process efficient and lightweight.

**Here’s how it works in simple terms:**  
- The Problem:  
  Fine-tuning a huge model like Stable Diffusion to recognize or replicate new styles or concepts (e.g., making it draw in the style of a specific artist or recognize unique objects) usually requires a lot of computational power and storage.

**The LoRA Solution:**  
- Instead of tweaking all the internal parameters of the Generative AI model, LoRA focuses only on a small subset of them. Think of it as adding a "style filter" to the model that can be applied or removed as needed.  
  It reduces the complexity by breaking down large changes into smaller, simpler steps.  
  These smaller steps don’t interfere with the original model, meaning you don’t lose the model’s core abilities.  

**Why it’s Cool:**
- Efficient: It uses way less memory and is faster than traditional fine-tuning methods.
- Flexible: You can train multiple LoRA "filters" for different styles or concepts and swap them in and out without modifying the base model.
- Compatible: LoRA modules can be shared or reused easily, so artists and developers can collaborate or try out others’ custom styles.

**Example Use Case**  
- Say you want to teach Generative AI models to draw in the style of a fictional artist.  
  You can train a LoRA on a handful of sample images in that style.  
  Once trained, the LoRA module acts like a plug-in—you just load it into Generative AI models, and the model starts generating images in that style!

In short, LoRA makes it easy to teach models new tricks without overwhelming your computer or altering the original model. It’s a user-friendly way to get customized results!  

## LoRA Types

There are many LoRA types, here are some of the most common ones: LoRA, DoRA, LoCon, HaDa, gLoRA, LoKR, LyCoris  
They vary in:
- Which model components are being trained. Typically UNET, but can be TE as well
- Which layers of the model are being trained. Each LoRA type trains different layers of the model
- Math algorithm to extrach LoRA weights for the specific trained layers

!!! warning

    LoRA must always match base model used for its training  
    For example, you cannot use SD1.5 LoRA with SD-XL model  

!!! warning

    SD.Next attempts to automatically detect and apply the correct LoRA type.  
    However, new LoRA types are popping up all the time
    If you find LoRA that is not compatible, please report it so we can add support for it.  

## How to use?

- Using UI: go to the networks tab and go to the lora's and select the lora you want and it will be added to the prompt.
- Manually: you can also add the lora manually by adding `<lora:lora_name:strength>` to the prompt and then selecting the lora you want to use.

### Trigger words

Some (not all) LoRAs associate specific words during training so same words can be used to trigger specific behavior from the LoRA.  
SD.Next displays these trigger words in the UI -> Networks -> LoRA, but they can also be used manually in the prompt.  

You can combine any number of LoRAs in a single prompt to get the desired output.  

!!! tip
  
    If you want to automatically apply trigger words/tags to prompt, you can use `auto-apply` feature in *"Settings -> Networks"*  

!!! tip

    You can change the strength of the lora by changing the number `<lora:name:x.x>` to the desired number  

!!! tip

    If you're combining multiple LoRAs, you can also "export" that as a single lora via *"Models -> Extract LoRA"*  

### Advanced

#### Component weights

Typically `:strength` is applied uniformly for all components of the LoRA.  
However, you can also specify individual component weights by adding `:comp=x.x` to the LoRA tag.  
Example:  `<lora:test_lora:te=0.5:unet=1.5>`  

#### Block weights

Instead of using simple `:strength`, you can specify individual block weights for LoRA by adding `:in=x.x:mid=y.y:out=z.z` to the LoRA tag.  
Example `<lora:test_lora:1.0:in=0:mid=1:out=0>`  

#### Stepwise weights

LoRA can also be applied will full per-step control by adding step-specific instuctions to the LoRA tag.  
Example: `<lora:test_lora:te=0.1@1,0.6@6>`  
Would mean apply LoRA to text-encoder with strength 0.1 on step 1 and then switch to strength 0.6 on step 6.  
#### Alternative loader

SD.Next actually contains two separate implementations for LORA:
- native implementation: used for most LoRAs
- `diffuser` implementation: used as a fallback for models where native implementation does not exist

Usage of `diffusers` implementation can be forced in *"Settings -> Networks"*
