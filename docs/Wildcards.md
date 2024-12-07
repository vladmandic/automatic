# Wildcards in Prompts

Wildcards are placeholders in the prompt text that are replaced with a random value from a list of choices which allows for more variety in the prompts generated.  

- SD.Next supports standard file-based wildcards in prompts.
- Wildcard support is enabled by default, enabled by default, can be disabled in *settings -> extra networks* if you want to use 3rd party extension instead of SD.Next built-in support.  
- Wildcards folder is set in *settings -> system paths*, default is `models\wildcards`

## How does it work?

*TL;DR*: string `"__abc__"` in prompt is matched to a file `abc.txt` inside wildcards folder

The prompt syntax for wildcards is:

> a woman wearing a ```__color__``` dress

In the wildcards folder, create file `color.txt` and add multiple choices with one choice per line:

    red
    green
    blue

## Tips

- Wildcards can be used in both positive and negative prompts  
- Prompt can have any number of wildcards
  > a woman wearing a ```__color__``` dress and a ```__shape__``` hat
- Wildcards can be nested  
  Line inside wildcard file can also have a wildcard referrring to another wildcard, etc.  
- Supports filename-only and path-based wildcards with full subfolder support  
  If wildcard is refered as `__color__` then it will look for file `color.txt` in wildcards folder and *any* subfoldrer  
  If wildcard is refered as `__nsp/color__` then it will look for `color.txt` *only* in `nsp` folder inside wildcards folder  
- Wildcards files can be in one-choice per line or multiple choices per line separated by `|` format  

### Validation

Wildcard matches and replacements will be logged in the standard log with debug level:

> DEBUG    Wildcards apply: wildcard="color" choice="Yellow" file="models/wildcards/my-variations/color.txt" choices=930
