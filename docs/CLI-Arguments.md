# Command Line Arguments

!!! tip

    All command line arguments can also be set as environment flags, for example `--debug` is equivalent to `SD_DEBUG=True`  
    All options listed here are available as arguments to use from the command line or as environment variables, there's no need to do both.

## List

> webui --help

    Configuration:
      --backend {original,diffusers}                     force model pipeline type
      --config CONFIG                                    Use specific server configuration file, default: config.json
      --ui-config UI_CONFIG                              Use specific UI configuration file, default: ui-config.json
      --medvram                                          Split model stages and keep only active part in VRAM, default: False
      --lowvram                                          Split model components and keep only active part in VRAM, default: False
      --freeze                                           Disable editing settings

    Paths:
      --ckpt CKPT                                        Path to model checkpoint to load immediately, default: None
      --data-dir DATA_DIR                                Base path where all user data is stored, default:
      --models-dir MODELS_DIR                            Base path where all models are stored, default: models

    Diagnostics:
      --no-hashing                                       Disable hashing of checkpoints, default: False
      --no-metadata                                      Disable reading of metadata from models, default: False
      --disable-queue                                    Disable queues, default: False
      --device-id DEVICE_ID                              Select the default CUDA device to use, default: None

    HTTP:
      --server-name SERVER_NAME                          Sets hostname of server, default: None
      --tls-keyfile TLS_KEYFILE                          Enable TLS and specify key file, default: None
      --tls-certfile TLS_CERTFILE                        Enable TLS and specify cert file, default: None
      --tls-selfsign                                     Enable TLS with self-signed certificates, default: False
      --cors-origins CORS_ORIGINS                        Allowed CORS origins as comma-separated list, default: None
      --cors-regex CORS_REGEX                            Allowed CORS origins as regular expression, default: None
      --subpath SUBPATH                                  Customize the URL subpath for usage with reverse proxy
      --autolaunch                                       Open the UI URL in the system's default browser upon launch
      --auth AUTH                                        Set access authentication like "user:pwd,user:pwd""
      --auth-file AUTH_FILE                              Set access authentication using file, default: None
      --api-only                                         Run in API only mode without starting UI
      --allowed-paths ALLOWED_PATHS [ALLOWED_PATHS ...]  add additional paths to paths allowed for web access
      --share                                            Enable UI accessible through Gradio site, default: False
      --insecure                                         Enable extensions tab regardless of other options, default: False
      --listen                                           Launch web server using public IP address, default: False
      --port PORT                                        Launch web server with given server port, default: 7860

    Setup:
      --reset                                            Reset main repository to latest version, default: False
      --upgrade, --update                                Upgrade main repository to latest version, default: False
      --requirements                                     Force re-check of requirements, default: False
      --reinstall                                        Force reinstallation of all requirements, default: False
      --uv                                               Use uv instead of pip to install the packages

    Startup:
      --quick                                            Bypass version checks, default: False
      --skip-requirements                                Skips checking and installing requirements, default: False
      --skip-extensions                                  Skips running individual extension installers, default: False
      --skip-git                                         Skips running all GIT operations, default: False
      --skip-torch                                       Skips running Torch checks, default: False
      --skip-all                                         Skips running all checks, default: False
      --skip-env                                         Skips setting of env variables during startup, default: False

    Compute Engine:
      --use-directml                                     Use DirectML if no compatible GPU is detected, default: False
      --use-openvino                                     Use Intel OpenVINO backend, default: False
      --use-ipex                                         Force use Intel OneAPI XPU backend, default: False
      --use-cuda                                         Force use nVidia CUDA backend, default: False
      --use-rocm                                         Force use AMD ROCm backend, default: False
      --use-zluda                                        Force use ZLUDA, AMD GPUs only, default: False
      --use-xformers                                     Force use xFormers cross-optimization, default: False

    Diagnostics:
      --safe                                             Run in safe mode with no user extensions
      --experimental                                     Allow unsupported versions of libraries, default: False
      --test                                             Run test only and exit
      --version                                          Print version information
      --ignore                                           Ignore any errors and attempt to continue

    Logging:
      --log LOG                                          Set log file, default: None
      --debug                                            Run installer with debug logging, default: False
      --profile                                          Run profiler, default: False
      --docs                                             Mount API docs, default: False
      --api-log                                          Enable logging of all API requests, default: False

## Details

### General Options

- `--config`: Specify the server configuration file. This option allows you to use a specific server configuration file. The default is set to `<path to data>/config.json`. You can customize this by providing a different file path or by setting the environment variable `SD_CONFIG`.
- `--ui-config`: Specify the UI configuration file. This option allows you to use a specific UI configuration file. The default is set to `<path to data>/ui-config.json`. You can customize this by providing a different file path or by setting the environment variable `SD_UICONFIG`.
- `--autolaunch`: Open the UI URL in the system's default browser upon launch. Enabling this flag (`True`) automatically opens the UI URL in the system's default browser upon launch. The default is `False`, and you can set it to `True` using the environment variable `SD_AUTOLAUNCH`.
- `--backend`: Force model pipeline type. Enabling this option allows you to force a specific model pipeline type. The choices are ['original', 'diffusers']. The default is `Original`, and you can set a custom backend by providing the argument value or by setting the environment variable `SD_BACKEND`.
- `--upgrade`: Upgrade the main repository to the latest version. Use this option when you want to ensure that you are using the most recent version of the application. Enabling this flag (`True`) upgrades the main repository to the latest version.  The default is `False`, and you can set it to `True` using the environment variable `SD_UPGRADE`.  
**This is relatively safe to use with master branch. Dev branch is another story, be wary.**
- `--debug`: Run SDNext with debug logging to the console. Enabling this flag (`True`) runs SDNext with debug logging. The default is `False`, and you can set it to `True` using the environment variable `SD_DEBUG`.  
**The sdnext.log files always show debug output for troubleshooting purposes.**
- `--test`: Run the tests only and exit. This is useful for checking the integrity of the application without starting the actual service. Also useful for doing installation or reinstallation activities. Enabling this flag (`True`) runs the application in test mode, executing tests and exiting without launching the full application.  The default is `False`, and you can set it to `True` using the environment variable `SD_TEST`.  
**Be careful setting this option as an environment variable as it will only test on every startup, you will never get inside SDNext.**  

### Skip Options

- `--quick`: Run with startup sequence only, does not check requirements, extensions, git, or torch tests. Enabling this flag (`True`) runs with the startup sequence only. The default is `False`, and you can set it to `True` using the environment variable `SD_QUICK`.
- `--skip-requirements`: Skips checking and installing requirements. Enabling this flag (`True`) during setup skips checking and installing requirements. The default is `False`, and you can set it to `True` using the environment variable `SD_SKIPREQUIREMENTS`.
- `--skip-extensions`: Skips running individual extension installers. Enabling this flag (`True`) during setup skips running individual extension installers. The default is `False`, and you can set it to `True` using the environment variable `SD_SKIPEXTENSION`.
- `--skip-git`: Skips running all GIT operations. Enabling this flag (`True`) during setup skips running all GIT operations. The default is `False`, and you can set it to `True` using the environment variable `SD_SKIPGIT`.
- `--skip-torch`: Skips running Torch checks. Enabling this flag (`True`) during setup skips running Torch checks. The default is `False`, and you can set it to `True` using the environment variable `SD_SKIPTORCH`.
- `--skip-env`: Skips setting env variables. Enabling this flag (`True`) during setup skips setting of any and all env variables used for tuning. The default is `False`, and you can set it to `True` using the environment variable `SD_SKIPENV`.

### Memory Management

- `--medvram`: Split model stages and keep only the active part in VRAM. Enabling this flag (`True`) allows the application to split model stages, conserving GPU memory by keeping only the active part in VRAM. The default is `False`, and you can set it to `True` using the environment variable `SD_MEDVRAM`.
- `--lowvram`: Split model components and keep only the active part in VRAM. Enabling this flag (`True`) allows the application to split model components. The default is `False`, and you can set it to `True` using the environment variable `SD_LOWVRAM`.

### Hardware Backends (Use During Installation or `--reinstall` operations)

- `--use-directml`: Use DirectML if no compatible GPU is detected. Enabling this flag (`True`) allows the use of DirectML if no compatible GPU is detected. The default is `False`, and you can set it to `True` using the environment variable `SD_USEDIRECTML`.
- `--use-openvino`: Use Intel OpenVINO backend. Enabling this flag (`True`) allows the use of the Intel OpenVINO backend. The default is `False`, and you can set it to `True` using the environment variable `SD_USEOPENVINO`.
- `--use-ipex`: Force use Intel OneAPI XPU backend. Enabling this flag (`True`) forces the use of the Intel OneAPI XPU backend. The default is `False`, and you can set it to `True` using the environment variable `SD_USEIPEX`.
- `--use-cuda`: Force use NVIDIA CUDA backend. Enabling this flag (`True`) forces the use of the NVIDIA CUDA backend. The default is `False`, and you can set it to `True` using the environment variable `SD_USECUDA`.
- `--use-rocm`: Force use AMD ROCm backend. Enabling this flag (`True`) forces the use of the AMD ROCm backend. The default is `False`, and you can set it to `True` using the environment variable `SD_USEROCM`.
- `--use-zluda`: Force use [ZLUDA](https://github.com/vosen/ZLUDA) backend. Enabling this flag (`True`) forces the use of the AMD ROCm backend wrapped with ZLUDA. The default is `False` and you can set it to `True` using the environment variable `SD_USEZLUDA`. You should not have `torch` installed before enabling this flag.

#### IPEX Environment Variables

- `DISABLE_IPEXRUN`: Disable ipexrun for Intel ARC. If you encounter memory pointer or SIGSEGV-related errors on Linux, you can try setting this environment variable to `True` to disable ipexrun.
- `IPEX_SDPA_SLICE_TRIGGER_RATE`: Specify when dynamic attention slicing for Scaled Dot Product Attention should get triggered for Intel ARC. This environment variable allows you to set the trigger rate in gigabytes (GB). The default is `6` GB.
- `IPEX_ATTENTION_SLICE_RATE`: Specify the dynamic attention slicing rate for Intel ARC. This environment variable allows you to set the slicing rate in gigabytes (GB). The default is `4` GB.
- `IPEX_FORCE_ATTENTION_SLICE`: Force use dynamic attention slicing even if the GPU supports 64 bit. Useful with Intel Data Center GPU MAX series.

### Pathing

- `--log`: Set log file name and path. This argument allows you to set the log filename and location. The default is `sdnext.log` in the base directory, and you can set it with the environment variable `SD_LOG`.
- `--ckpt`: Path to the model checkpoint to load immediately. This option allows you to specify the path to a model checkpoint for immediate loading. The default is `None`, and you can set a custom path by providing the argument value or by setting the environment variable `SD_MODEL`.
- `--vae`: Path to the VAE checkpoint to load immediately. This option allows you to specify the path to a VAE checkpoint for immediate loading. The default is `None`, and you can set a custom path by providing the argument value or by setting the environment variable `SD_VAE`.
- `--data-dir`: Base path where all user data is stored. You can set the base path where all user data is stored using this option. The default is an empty string (`''`). Customize this by providing a different path or by setting the environment variable `SD_DATADIR`.
- `--models-dir`: Base path where all models are stored. This option sets the base path where all models are stored. The default is `'models'`. Customize this by providing a different path or by setting the environment variable `SD_MODELSDIR`.

### Troubleshooting Options  

**Also see [Troubleshooting Wiki page](https://github.com/vladmandic/automatic/wiki/Troubleshooting)**

- `--safe`: Run in safe mode with no user extensions. Safe mode can be useful when troubleshooting or when you want to restrict the execution of potentially unsafe user-provided code. Enabling this flag (`True`) runs the application in safe mode, disabling user extensions.  The default is `False`, and you can set it to `True` using the environment variable `SD_SAFE`.  
**Try this first.**
- `--requirements`: Force re-check of python (pip) package requirements and installs any that are not up-to-date. Enabling this flag (`True`) forces a re-check of requirements. The default is `False`, and you can set it to `True` using the environment variable `SD_REQUIREMENTS`.  
**Try this second.**
- `--reinstall`: Force reinstallation of all requirements. Use this option when you want to ensure that all dependencies are freshly installed, potentially resolving any issues related to outdated or corrupted installations. Also useful to change hardware backends, such as to OpenVINO or DML. Enabling this flag (`True`) forces the reinstallation of all requirements. The default is `False`, and you can set it to `True` using the environment variable `SD_REINSTALL`.  
**Try this third.**  
**Be careful setting this option as an environment variable as it will reinstall on every startup.**
- `--reset`: Reset main repository to latest version. Enabling this flag (`True`) resets the main repository to the latest version. The default is `False`, and you can set it to `True` using the environment variable `SD_RESET`.  
**This is more serious with consequences, but you can try this fourth.**  
**Be careful setting this option as an environment variable as it will reset on every startup.**
- `--experimental`: Allow unsupported versions of libraries. This is useful for testing or trying out features that may not be officially supported yet. Enabling this flag (`True`) allows the application to use unsupported versions of libraries.  The default is `False`, and you can set it to `True` using the environment variable `SD_EXPERIMENTAL`.  
**Be careful setting this option as an environment variable as it can easily cause issues.**
- `--ignore`: Try to ignore any errors and attempt to continue. This can be useful in scenarios where certain errors are known and can be safely bypassed. Enabling this flag (`True`) instructs the application to ignore any errors encountered during the setup process and attempt to continue. The default is `False`, and you can set it to `True` using the environment variable `SD_IGNORE`.

### Advanced Options

- `--device-id`: Select the default CUDA device to use. This option allows you to select the default CUDA device for GPU operations. The default is `None`, and you can set a custom device ID by providing the argument value or by setting the environment variable `SD_DEVICEID`.
- `--use-xformers`: Forces the installation and use of xFormers cross-optimization. Enabling this flag (`True`) forces the installation and use of xFormers. The default is `False`, and you can set it to `True` using the environment variable `SD_USEXFORMERS`.
- `--no-hashing`: Disable hashing of checkpoints. Enabling this flag (`True`) disables the hashing of checkpoints. The default is `False`, and you can set it to `True` using the environment variable `SD_NOHASHING`.
- `--no-metadata`: Disable reading of metadata from models. Enabling this flag (`True`) disables the reading of metadata from models. The default is `False`, and you can set it to `True` using the environment variable `SD_NOMETADATA`.
- `--profile`: Run profiler. Enabling this flag (`True`) runs the profiler. The default is `False`, and you can set it to `True` using the environment variable `SD_PROFILE`.
- `--disable-queue`: Disable queues. Enabling this flag (`True`) disables the use of queues. The default is `False`, and you can set it to `True` using the environment variable `SD_DISABLEQUEUE`.
- `--allow-code`: Allow custom script execution. This is useful for scenarios where users may want to run their own code. Enabling this flag (`True`) allows the execution of custom scripts.  The default is `False`, and you can set it to `True` using the environment variable `SD_ALLOWCODE`.
- `--use-cpu`: This option forces the use of CPU for specified modules. You can provide a list of modules as arguments. The default is an empty list (`[]`). Customize this by providing module names or by setting the environment variable `SD_USECPU`.  
**Not advised, legacy code with poor performance.**
- `--freeze`: Disable editing settings. This is useful to lock down configurations. Enabling this flag (`True`) prevents editing of settings.  The default is `False`, and you can set it to `True` using the environment variable `SD_FREEZE`.

### API

- `--docs`: Mount API docs at /docs i.e., `https://127.0.0.1/docs`. Enabling this flag (`True`) mounts API documentation at the `/docs` endpoint. The default is `False`, and you can set it to `True` using the environment variable `SD_DOCS`.
- `--api-only`: Run in API only mode without starting UI. Enabling this flag (`True`) runs the application in API-only mode without starting the UI. The default is `False`, and you can set it to `True` using the environment variable `SD_APIONLY`.
- `--api-log`: Enable logging of all API requests. Enabling this flag (`True`) logs all API requests. The default is `False`, and you can set it to `True` using the environment variable `SD_APILOG`.

### Networking

- `--share`: Enable UI to be accessible through the Gradio site. This is useful for sharing your SDNext with others. Enabling this flag (`True`) allows the UI to be accessible through the Gradio site.  The default is `False`, and you can set it to `True` using the environment variable `SD_SHARE`.  
**The shared URL will be in your console log.**
- `--insecure`: Enable extensions tab regardless of other options. This should only be used when you want local network or web accessible control of your extensions, **potentially dangerous on the web.** Enabling this flag (`True`) allows the extensions tab to be enabled regardless of other specified options. The default is `False`, and you can set it to `True` using the environment variable `SD_INSECURE`.  
**Use with care when using `--share`.**
- `--listen`: Launch web server to be accessible from local network. Enabling this flag (`True`) allows the web server to launch for use on your own network. The default is `False`, and you can set it to `True` using the environment variable `SD_LISTEN`.
- `--auth`: Set access authentication like "user:pwd,user:pwd". This option allows you to set access authentication with a specified username and password combination. The default is `None`, and you can set a custom authentication string by providing the argument value or by setting the environment variable `SD_AUTH`.
- `--auth-file`: Set access authentication using file. This option allows you to set access authentication using a file. The default is `None`, and you can set a custom file path by providing the argument value or by setting the environment variable `SD_AUTHFILE`.
- `--server-name`: Sets hostname of server. This option sets the hostname of the server. The default is `None`, and you can set a custom server name by providing the argument value or by setting the environment variable `SD_SERVERNAME`.
- `--subpath`: Customize the URL subpath for usage with reverse proxy. This option allows you to customize the URL subpath for usage with a reverse proxy. The default is `None`, and you can set a custom subpath by providing the argument value or by setting the environment variable `SD_SUBPATH`.
- `--cors-origins`: Allowed CORS origins as comma-separated list. This option sets the allowed CORS origins as a comma-separated list. The default is `None`, and you can set custom origins by providing the argument value or by setting the environment variable `SD_CORSORIGINS`.
- `--cors-regex`: Allowed CORS origins as regular expression. This option sets the allowed CORS origins as a regular expression. The default is `None`, and you can set a custom regular expression by providing the argument value or by setting the environment variable `SD_CORSREGEX`.
- `--tls-keyfile`: Enable TLS and specify key file. This option enables TLS (Transport Layer Security) and specifies the key file. The default is `None`, and you can set a custom key file path by providing the argument value or by setting the environment variable `SD_TLSKEYFILE`.
- `--tls-certfile`: Enable TLS and specify cert file. This option enables TLS (Transport Layer Security) and specifies the certificate file. The default is `None`, and you can set a custom certificate file path by providing the argument value or by setting the environment variable `SD_TLSCERTFILE`.
- `--tls-selfsign`: Enable TLS with self-signed certificates. Enabling this flag (`True`) enables TLS with self-signed certificates. The default is `False`, and you can set it to `True` using the environment variable `SD_TLSSELFSIGN`.

#### CORS in depth

These CORS options are important for controlling which domains are permitted to access the resources of your server. It helps in enhancing the security of your web application by preventing unauthorized cross-origin requests. When configuring these options, ensure that you only allow origins that you trust to interact with your server, as allowing any origin (`*`) can introduce security vulnerabilities. Always specify the origins explicitly or using a secure regular expression pattern.

- `--cors-origins`: Allowed CORS origins as a comma-separated list. CORS is a security feature implemented by web browsers that restricts web pages from making requests to a different domain than the one that served the web page. This option allows you to specify a list of origins (domains) that are allowed to access resources on your server. For example, if your application is hosted on `http://example.com`, and you want to allow access from `http://client.example.com`, you would set `--cors-origins http://client.example.com`.  
**For local network use, often with other tools, you can just set `--cors-origins *`.**
- `--cors-regex`: Allowed CORS origins as a regular expression. This option provides more flexibility by allowing you to specify CORS origins using a regular expression. The regular expression should match the origin(s) you want to allow. This is useful when you have a dynamic set of origins that follow a certain pattern. For instance, if you want to allow any subdomain under `example.com`, you could set `--cors-regex "^https?://[a-z0-9-]+\.example\.com$"`.

#### TLS in depth

When setting up TLS, it's important to use valid certificates from a trusted CA in a production environment to ensure secure and encrypted communication. In a development or testing environment, self-signed certificates can be used, but users should be cautious when accessing the application as browsers may show security warnings due to the self-signed nature. Always ensure that your TLS setup meets security best practices.

- `--tls-keyfile`: Enable TLS and specify the key file. TLS is a cryptographic protocol that ensures the secure transmission of data over a network. Enabling TLS in your application secures the communication between the client and the server. The `--tls-keyfile` option allows you to specify the path to the private key file used for encryption. This private key file should be in PEM format.

  Example:

  ```bash
  --tls-keyfile /path/to/private-key.pem
  ```

- `--tls-certfile`: Enable TLS and specify the cert file. Along with the private key, you need to specify the TLS certificate file. The certificate file contains the public key and information about the server. It should be in PEM format.

  Example:

  ```bash
  --tls-certfile /path/to/certificate.pem
  ```

- `--tls-selfsign`: Enable TLS with self-signed certificates. If you don't have a certificate issued by a Certificate Authority (CA), you can enable this option to create self-signed certificates. Self-signed certificates are useful for development and testing but should not be used in production environments where security is crucial.

  Example:

  ```bash
  --tls-selfsign /path/to/self-signed-certificate.pem
  ```
