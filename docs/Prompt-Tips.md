# Prompt Tips

> [!NOTE] Different models have different prompting best practices. When in doubt, refer to model specific notes on CivitiAI website

For example: standard **SD15/SDXL** models are prompted very differently vs **Pony**-derived models vs **SD35** or **Flux.1** derived models - it all comes down to which text enocder is used and how it was trained.  
For more details, see [Prompting](Prompting.md) with different models wiki page.

## Params

TL;DR: Tweak **steps**, **cfg scale** and **sampler** as results will vary depending on combination of all three  

- **Encoder**  
  Which text tokenizer to use, SD typically uses `CLiP`, but others can be substituted (`BERT`, `GPTx`, etc)  
- **Batch Size**  
  How many images to generate in parallel, limited by your VRAM  
- **Batch Count**  
  How many batches to run sequentially  
  So total number of images generated is batch size x batch count  
- **Seed**  
  Initializer for noise generator  
  Use same seed to have repeatable results, otherwise use random (-1)  
- **CFG Scale** (Classifer-Free-Guidance)  
  How close should diffusers follow prompt, 0 means none and 30 means exact  
  Best results are between 7 (creative) to 13 (realistic), but optimal value depends on your model, prompt, and parameters  
- **Width & Height**  
  SD 1.x was trained on 512x512, SD 2.x on 768x768, SDXL on 1024x1024, derivative fine-tunes may have different resolutions  
  So typically don't stray too far from those and instead use upscalers if high resolution is needed  
  However, changing aspect ratio can change composition of image (e.g. portrait vs landscape results in close-up vs more wide angle results)  
- **Steps**  
  Directly impacts performance  
  How many iterative denoising steps to run, low number can lead to non-converged results (denoising is not complete)  
  Sweet-spot depends on chosen sampler and settings, can be as low as 10 and as high as 100  
  Higher number of steps tends to increase output quality, except for non-converging (ancestral) samplers like "Euler a" which just keep modifying the picture to no end  
  At high step counts, many samplers converge to the same image as other samplers  

## Prompt Engineering

Know your model: different models were trained on different datasets, some may understand terms other models don't  

**Main groups**

- **Mediums**: best starting a prompt with it after specifying artist  
  Examples: *painting, photograph, drawing, sketch*
- **Flavors**: best left as separate token at the end of the prompt  
  Examples: *ray tracing, fine art, black and white, pixiv, artstation*
- **Movements**: best added to prompt with as keyword  
  Examples: *pop art, photorealism*  
- **Artists**: best starting a prompt with it  
  Examples: *greg rutkowski, artgerm, dc comics, picasso*  

**Modifiers**

- **Feel**: best near the end  
  Examples: *beautiful, sharp focus, 4k, hdr, high detailed, canon 5d*
- **Composition**: best at front, but only use if results don't fit  
  Examples: *1man, 1woman*

**Negative Prompt**

- Any keyword can be specified in a negative prompt as well
  Examples: *watermark*

**Advanced Prompt Modifiers**  

- Availability depends on implementation  
- Specify importance of specific words: E.g. using `(word:1.2)` makes the influence of `word` stronger, `(word:0.8)` makes it weaker  

**Advanced Prompt Modifiers**  

For original backend only:
- Alternate between words: `[word1|word2]` will alternate between `word1` and `word2` in every denoising step, blending the two concepts  
- Switch words during denoising: `[word1:word2:0.3]` will use `word1` for the first 30% of steps, then change it to `word2`  
- Force include multiple objects "AND"  

**Hints**

- Use either artists or movements  
  Using both may result in one overpowering the other, or in unexpected outcome  
- Select medium that fits artist  
  It helps model a lot to know which medium to use when styling  
- Add action after subject  
  Examples: portrait, standing, sitting
- Moving things to the front of prompt may increase its emphasis  
  Example: *cartoon drawing of a woman as pixar* vs *pixar drawing of a woman*
- Use both subject and scene keywords
  Example: *woman on a beach*

**Example**

> (composition) (artist) (medium) (subject) (action) (scene) (movement) (flavor) (feel)  
> 1woman greg rutkowski painting of a woman happy front portrait on a beach as photorealism, sharp focus, artstation
