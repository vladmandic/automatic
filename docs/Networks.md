# Networks

The Networks button under Generation Controls opens a convenient side menu, providing shortcuts to your:
- Models
- LoRas: see [LoRA](LoRA.md) for more information
- Styles: see [Styles](Styles.md) for more information
- Embeddings
- VAEs
- Latent history

## Reference models

In the models tab, you can find Reference models section  
This is the list of predefined models that can be immediately selected and used
Once reference model is selected, it will be automatically downloaded and saved in the models folder for future use  

!!! tip

    Reference models is recommended way to start with any base model

Reference models section can be hidden in *"Settings -> Networks"*  

## Latent history  

Near the completion of each generate workflow, its latents (raw data from model before final decoding) are added to latent history.  
Why? So they can be reused and reprocessed at will.  

For example, *text + refine + detailer* will generate 3 entries in latent history, one for each step.  
You can then **reprocess** any of these steps with different settings or even different models.

You can control the length of maintained latent history in *"Settings -> Execution"*
