# StableDiffusion Training Methods

## fine-tuning

- retrains parts of the hypernetwork with new data thus modifying original weights  
  requires large and precisely labelled dataset
- size is same as original model size, ~2-7gb
- verdict: prohibitive due to large dataset and effort required

## model merge

- combines weights from multiple models according to specified rules
- verdict: highly desired to create pre-set models for specific use-case

## textual inversion

- assign vector to a new concept with originally one vector per embedding, hacks to enable multi-vector embeddings  
  works by expanding vocabulary of a model, but majority of learned content is actually assembled from existing concepts  
  can be considered as a formula on which already learned weights should be combined to achieve learned concept  
- size 768/1024b per vector
- verdict: best currently viable short-term training solution

## aesthetic gradient

- uses low-precision trained embeddings to steer clip using classifier guidance  
  training is very cheap, but classifier guidance sloes down image generation  
  result is basic transfer of style from learned image to generated image  
- size is same as embedding
- origin: independent work
- verdict: inconsistent results with minimal value

## custom diffusion

- fine-tuning specific model matrices with textual inversion  
  similar speed and memory requirements to embedding training and supposedly gives better results in less steps
- size ~50mb
- origin: cmu
- verdict: possibly promising, requires further investigation, surprisingly low chatter on this topic

## hypernetwork

- similar to model fine-tuning, but adds small a small neural network that on-the-fly modifies weights of the last two layers of the main model  
  works like adaptive head that steers model in a learned direction so primary use-case is style transfer, not concept transfer
- size is limited to learned layers, ~100-200mb
- origin: leaked from novel.ai
- verdict: lower priority as concept transfer is more important than style transfer

## null-text inversion

- similar concept to textual inversion, but trains unconditional embedding that is used for classifier free guidance instead of text embedding  
  resulting embedding is apparently more detailed than standard textual embedding
- size is larger but comparable to textual inversion
- origin: google
- verdict: possibly promising, requires further investigation, but no working prototype as of yet

## clip inversion

- similar concept to textual inversion, but uses clip embedding instead of text embedding  
- size is same as textual inversion
- origin: google
- verdict: prohibitive due to requirement of specially fine-tuned model as a starting point

## dream artist

- variation on ti training where both positive and negative embeddings are created
- size is same as textual inversion
- origin: independent work
- verdict: skip for now as solution does not appear to be sufficiently maintained

## dreambooth

- similar to model fine-tuning except it adds information on top of model instead of forgetting/overwriting existing concepts  
- size is equal to original model size, ~2-7gb
- origin: google, but heavily modified by independent work
- verdict: prohibitive due to resulting size and requirement to load full model on-demand

## lora

- "low-rank adaptation of large language models"  
  injects trainable layers to steer cross attention layers  
  very flexible, but memory intensive so limited training opportunities on normal gpu  
  multiple incompatible implementations: should choose which implementation to use  
- size varies from ~5mb to full-model size, average ~150-300mb
- origin: microsoft
