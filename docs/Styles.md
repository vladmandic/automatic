# Styles

Styles are powerful feature in SD.Next that allow to apply various modifications to your generation parameters:
- Prompt: both postive and negative  
- Parameters: all generation parameters  
- Wildcards  

![image](https://github.com/user-attachments/assets/9b147274-24db-4dbc-a27a-75072a6d2aaa)

!!! note

    Styles wildcards are separate feature that standard [wildcards](Wildcards) which can also be used in parallel

Styles can be selected via Networks interface -> Styles  
or via shortcut combo-box control below generate buttons  

There can be any number of styles selected and each style will be applied in order they are selected.

Each style is a separate JSON file that can be edited manually or via UI.
Location of styles is specified in Settings -> System Paths -> Styles folder, default is `models\styles`  

!!! tip

    Button ↶ "*Apply selected style to prompt*" will apply currently selected styles to current prompt as-is and remove style from being applied during runtime

!!! tip

    Button ↷ "*Save current prompt to style*" will simply save current prompt to named style. Such style can be later edited for more fine-tuning  

## Migration

Old A1111 style concept was a flat file in CSV format and SD.Next supports migration of such styles to new JSON format  
Simply put a full path to the CSV file in *Settings -> System Paths -> Styles* and restart server at which point  
server will migrate found styles into individual JSON files.

## Prompt

Prompt specified in style will be either used to replace a placeholder `{prompt}` in the current prompt or if there is no placeholder, it will be appeneded at the end of the current prompt  

Example:
> "national geographic style photo shot on sony a7 camera"  

## Parameters

In addition to prompt, you can also specify any generation parameters in the style  
List of parameters is comma-separated and each parameter is a key-value pair indicated by colon `:`  
Recognized parameters are all parameters that can be typically found in image metadata  

Example:

> Sampler: Euler a, CFG scale: 6.5, Steps: 25, Width: 1440, Height: 720  

## Wildcards

Both prompt and parameters can be modified using wildcards section inside the style  
List of wildcards is comma-separated and each wildcard is a key-value pair indicated by `=`  
Multiple wildcards can be listed and separated by semi-colon `;`  

Example#1:
> Prompt: "a woman wearing a {color} dress"  
> Wildcard: "{color}=red, green, blue"  

Example#2:
> Prompt: "{style} a woman wearing a {color} dress"  
> Wildcard: "{style}=photo, sketch, painting; {color}=red, green, blue"  

Example#3:

> Parameters: "Size: {size}"  
> Wildcard: "{size}=1024x1024, 1024x768, 768x1024, 1280x720, 720x1280, 1536x640"  

## JSON

Structure of the style is a simple JSON object:

```json
{
  "name": "Cute Robot",
  "description": "This is a style of a random cute robot",
  "prompt": "photo of a cute {color} robot, walking {where} with {background} visible in background",
  "negative": "",
  "extra": "Size: {size}",
  "wildcards": "
    {color}=blue, red, rusty, silver, cyan;
    {where}=on alien planet, in rainforest, in the city street;
    {background}=rocks and mountains, moon and planets, spaceship, battle;
    {size}=1024x1024, 1280x720, 720x1280"
}
```

## Validation

Styles use will be logged in the standard log with debug level:

> DEBUG    Applying style: name="mine/Cute Robot" extra=[] skipped=[] reference=False  
> DEBUG    Wildcards applied: {'{color}': 'red', '{what}': 'water', '{background}': 'moon and planets'} path="/mnt/models/wildcards" type=style time=0.00  
