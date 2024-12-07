# Prompting with different models

This is not in-depth technical article, but instead intended to demystify the process of prompting with different models - and importantly - why.

Basically, effectiveness prompting depends on:
- Text encoder used in the model
- Dataset used to train the model
- Structure prompt and negative prompt

## Components

### Text encoder

- SD15 started with: [CLIP-ViT/L](https://huggingface.co/openai/clip-vit-large-patch14)
- SDXL adds second encoder: [OpenCLIP-ViT/G](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
- SD3 adds third (optional) encoder: [T5 Version 1.1](https://huggingface.co/google/t5-v1_1-xxl)

And other models are following similar approach - replace early simple encoders with only basic understanding of English language with more advanced models.
For example: [PixArt-Σ](https://pixart-alpha.github.io/PixArt-sigma-project/) and [Tencent HunyuanDiT](https://github.com/Tencent/HunyuanDiT)

Unfortunately, in StabilityAI models, text encoder results are concatenated one after each other so oldest CLIP-ViT/L still has biggest impact.

### Dataset

Nearly all modern models are trained on subset of [laion-5b](https://laion.ai/blog/laion-5b/) dataset.
Later fine-tuning introduces additional data, but that is not as important as the base dataset.

What differs is how that subset is processed and captioned:

## Models

### SD15

Dataset:
- Used small laion-5b subset with preexisting captions
- No major effort was put into processing or captioning other than what's in the subset

Result?
- Model that's has basic understanding of the world, but to use it effectively you need to "hunt" for the right keywords  
- Models are easily fine-tuned as nothing-forced-nothing-removed approach was used during training  
- **How to prompt**? Old-school prompting which put heavy emphasis on keywords and attention modifiers  
  
### SD21

Dataset:
- Trained on same dataset as SD15 and then fine-tuned on extended dataset with larger resolution  
- It was also censored in the final parts of training which introduced heavy bias  

Result?
- Its almost like model was "lobotomized".  
  E.g. For concept of "topless", its not like it just doesn't have sample data for it, it was "burned out" of the model.
  So to add concepts that model did not understand it takes massive effort, almost close to retraining. Fail.

### SDXL

Dataset:
- Used larger subset with extended captions and also diverse resolutions  
- Instead of censoring in the final stages, they simply pruned dataset used for training.

Result?
- Model that *knows-what-it-knows* and the rest can be added with fine-tuning.
- E.g. It knows what "topless" means, just doesn't have enough examples to develop it fully.  
- **How to prompt**? Extended captions and second text encoder mean that model can be prompted in a more natural way and extended use of keywords and attention-modifiers should be avoided.

- Extra note: **PonyXL** was extensively trained on heavily tagged dataset without natural language captions and as such it needs to be prompted differently - using tags and keywords instead of natural language.

### SD3

- Used even larger subset with even more diverse resolutions, but dataset itself was processed differently:
- Censored not only by removing images from dataset, but also by modifying them by censoring parts of the image
- Captioned extensively using LLM. Unfortunately, not ON-TOP of existing captions so it can augment them, but instead it REPLACED them - thus keywords already existing in dataset are not trained for at all.

Result?
- Model that *thinks-it-knows-everything* and now it's up to you to prove it wrong.  
  E.g., it knows what a topless person looks like, and its "certain" that nipples should be presented as blank.  
  Which means it would like take a massive effort to retrain what it learned in the wrong way.  
  I hope to be proven wrong, but this looks like a fail.
- **How to prompt**? Use of long LLM-generated captions means that model should be prompted using very descriptive language and completely stop using using styles, keywords and attention-modifiers. And since LLM generated captions do not include styles as we know them, we need to replace them with detailed descriptions - it's almost like we need to think like LLM to prompt it - how would LLM describe the image I'm trying to create?

## Prompting tips

### Negative prompt

When you add negative prompt, what happens is that its basically appended to prompt just using negative weights.
This makes model "steer away" from terms in negative prompt, but to do so it first has to INTRODUCE them to the context.

So by adding negative prompt, you're:
- Limiting the freedom of the model  
  Quite commonly this means that all faces will look similar without variance, etc.
- Making the model more prone to hallucinations
  If you're trying to steer away from something that doesn't exist in the first place, it might introduce opposite of what you're trying to do.

All-in-all, negative prompts are useful to steer model away from certain concepts, but should not be used as a *"general purpose long negative prompt"*

### Prompt attention

First, nothing wrong with wanting to add extra attention to certain parts of the prompt. But it should be used sparingly and only when needed and keep in mind that overall prompt should be balanced.

E.g, If your prompt has 10 words and you're raising attention to 5 of them, you're basically telling the model average weight of the prompt is massive. Common result? Overbaked images.

Prompt balance doesn't have to be perfect, but any prompt that has more than few of words with light attention modifiers is a red flag.  

Also, keep in mind that adding extremely strong attention modifiers such as `(((xyz)))` or `(xyz:1.5)` will make model completely loose concept of prompt as a whole.
