# Gated Models

## Huggingface Login

Access to some models is gated by vendor and in those cases, you need to request access to model from the vendor.  
For this you need to have a valid Huggingface account: [Login](https://huggingface.co/login) or [Sign Up](https://huggingface.co/join)  

Huggingface login and/or access token is not required for non-gated models  

### Create Token

*Note*: This is a one-time operation as same access token is used for all gated models.

Once you are logged in, create access token that an external application such as **SD.Next** can use to access **Huggingface** on your behalf:

Go to: *Huggingface -> Profile -> Settings -> Access Token -> Create new token*  
Or use [this link](https://huggingface.co/settings/tokens/new?tokenType=read)  

- Token type: READ  
  Do not use fine-grained to avoid complications  
  Name is your choice  
- Create token  
  Copy the token and store it in a safe place  

### Add Token to SD.Next

Go to: *SD.Next -> System -> Settings -> Diffusers*  
- Paste the token in the Huggingface Token field  

## Requesting Access

*Note*: Requesting access must be done on individual per-model case

Requesting access can be in the form of simply accepting vendors terms of service or filling a form to get access to the model or requesting access and waiting for approval.  
In all cases, you need to go to model page on Huggingface and follow instruction.  

Examples: [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev), [SD3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)

Once you have access, you can use the model in SD.Next as usual
