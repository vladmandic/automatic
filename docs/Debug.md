# Debug

To run **SD.Next** in debug mode, start it with `--debug` flag  
This has no overhead and can be safely used in daily operations as it just prints additional information to logs  

Example:
  > webui.bat --debug  
  > webui.sh --debug  

## Extra Debug

Some debug information would be too much for regular use, so it can be enabled by use of environment variables:

### Install & load

- `SD_INSTALL_DEBUG`: report detailed information related to packages installation  
- `SD_PATH_DEBUG`: report all used paths as they are parsed
- `SD_SCRIPT_DEBUG`: increase verbosity of script and extension load and execution
- `SD_MOVE_DEBUG`: trace all model moves from and to cpu/gpu  
- `SD_EXT_DEBUG`: trace extensions load/install/update operations  
- `SD_LOAD_DEBUG`: report all model loading operations as they happen

### Core processing

- `SD_PROCESS_DEBUG`: print detailed processing information
- `SD_DIFFUSER_DEBUG`: increase verbosity of diffusers processing
- `SD_LDM_DEBUG`: increase verbosity of LDM processing
- `SD_CONTROL_DEBUG`: report all debug information related to control module

### Extra networks

- `SD_EN_DEBUG`: report all extra networks operations as they happen
- `SD_LORA_DEBUG`: increase verbosity of LoRA loading and execution

### Other

- `SD_PASTE_DEBUG`: report all params paste and parse operations as they happen
- `SD_HDR_DEBUG`: print HDR processing information
- `SD_PROMPT_DEBUG`: print all prompt parsing and encoding information
- `SD_SAMPLER_DEBUG`: report all possible sampler settings for selected sampler
- `SD_STEPS_DEBUG`: report calculations done to scheduler steps
- `SD_VAE_DEBUG`: report details on all VAE operations
- `SD_MASK_DEBUG`: reported detailed information on image masking operations as they happen  
- `SD_DOWNLOAD_DEBUG`: report detailed information on model download operations as they happen
- `SD_CALLBACK_DEBUG`: report each step as it executes with full details  
- `SD_BROWSER_DEBUG`: report all gallery operations as they happen
- `SD_NAMEGEN_DEBUG`: report all filename generation operations as they happen

Example *Windows*:
  > set SD_PROCESS_DEBUG=true  
  > webui.bat --debug  

Example *Linux*:
  > export SD_PROCESS_DEBUG=true  
  > webui.sh --debug  

Additional information enabled via env variables will show in log with level `TRACE`

## Profiling

To run **SD.Next** in profiling mode, start it with `--profile` flag  
This does have overhead, both on processing and memory side, so its not recommended for daily use  
SD.Next will collect profiling information from both Python, Torch and CUDA and print it upon completion of specific operations  

Example:
  > webui.bat --debug --profile  
  > webui.sh --debug --profile
