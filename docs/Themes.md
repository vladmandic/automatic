# Themes

SD.Next supports two native theme engines plus option to disable it completely to use external themes:

## Set Themes

**Set theme via UI**: *settings -> user interface*
- theme type: **modern**, **standard**, **none**  
  *note*: **none** disables native theme engine and is used  
  for *gradio* built-in themes, *huggingface* 3rd party themes and custom *extension* based themes  
- theme name  
- theme mode  
  to force light/dark or leave it as os-default (auto)  

**Set theme via CLI**: `--theme theme-type/theme-name`  
- theme `default` defaults to **standard/black-teal**
- optional theme types: **standard**, **modern**, **gradio**, **huggingface**  
  if theme type is not specified, it will default to **standard**  
- if theme name is not specified, it will default to:  
  - **default** for modern
  - **black-teal** for standard
  - **gradio/default** for gradio
  - **huggingface/none** for huggingface
- theme param can additionally be used to enable to specific theme extension:
  - **lobe**
  - **cozy-next**

Selected theme type and name will be shown in the log on startup  
example:

    11:41:37-649897 DEBUG    UI themes available: type=Standard themes=12
    11:41:37-650510 INFO     UI theme: type=Standard name="black-teal" style=Auto
    11:41:37-651747 DEBUG    UI theme: css="/home/vlado/dev/sdnext/javascript/black-teal.css" base="sdnext.css" user="None"

or:

    11:42:42-946642 DEBUG    UI themes available: type=Modern themes=22
    11:42:42-947313 INFO     UI theme: type=Modern name="sdxl_alpha" style=Auto
    11:42:42-948546 DEBUG    UI theme: css="extensions-builtin/sdnext-modernui/themes/sdxl_alpha.css" base="base.css" user="None"

### Switching themes

Once you set theme type, themes of type **standard** and **modern** can be switched on the fly without restarts  

## Creating Custom Themes

See [User Themes](Theme-User) for details on creating custom themes  

## Available Themes

### Standard Themes

SD.Next comes with number of built-in themes:

- Black teal (default) ![](/html/black-teal.jpg)
- Light teal ![](/html/light-teal.jpg)
- Simple dark ![](/html/simple-dark.jpg)
- Simple light ![](/html/simple-light.jpg)
- Black orange ![](/html/black-orange.jpg)

Following community created themes are included in SD.Next:

- Invoked ![](/html/invoked.jpg)
- Amethisyt nightfall ![](/html/amethyst-nightfall.jpg)
- Emerald paradise ![](/html/emerald-paradise.jpg)
- Midnight barbie ![](/html/midnight-barbie.jpg)
- Orchid dreams ![](/html/orchid-dreams.jpg)
- Timeless beige ![](/html/timeless-beige.jpg)

### Modern Themes

> [!IMPORTANT]  
> Any issues related to modern ui should be reported at: <https://github.com/BinaryQuantumSoul/sdnext-modernui/issues>

### Gradio Themes

- Gradio default ![](/html/gradio-default.jpg)
- Gradio base ![](/html/gradio-base.jpg)
- Gradio soft ![](/html/gradio-soft.jpg)
- Gradio glass ![](/html/gradio-glass.jpg)
- Gradio monochrome ![](/html/gradio-monochrome.jpg)

Use of Gradio themes disables built-in theme engine and uses Gradio theme engine instead  
Gradio themes are not optimized for SDNext and will likely cause some UI components to look out of place  

### Huggingface Themes

When you refresh list of themes using System -> Settings -> User Interface -> Themes -> Refresh  
SD.Next will download list of 3rd party Gradio themes hosted on Huggingface  

Note that formatting of UI components in that case depends on theme itself and is outside of SD.Next control  

### Extensions

SDNext also supports custom themes via extensions  
Currently listed are **cozy-next** and **lobe** themes, however those themes are not updated for recent SDNext releases - please contact extension authors for updates  
