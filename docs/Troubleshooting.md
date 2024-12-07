# Troubleshooting Common Issues

If you're having issues with SD.Next, please follow these steps designed to help weed out known issues first.  
All users should do Steps #1 and #2 regardless of having a problem or not.

## 1. CLI Arguments

You should familiarize yourself with all available CLI arguments by typing `webui.bat (or .sh) --help`, which  
will present a full list of them to you. There are likely options you have at your disposal that you are unaware of.

## 2. UI Config

If your `ui-config.json` file is larger than a few (1-20) kb, delete it. The way the UI config file works has  
changed, now it only saves the differences between SD.Next's defaults and what you have set rather than the older  
bloated file that contained everything. Issues arise from new settings and defaults being overridden by the existing  
old settings that are no longer valid, this can even lead to non-functional buttons.

## 3. Config

Often many issues are cleared up by simply deleting the config.json file and letting SD.Next generate a new one. However  
this is destructive and annoying because you have to set all of your personal preferences again, including model/image paths.  
Instead we recommend simply renaming `config.json` to `config-backup.json`. This way the system will generate a new file when  
you restart SD.Next, while also preserving your paths and settings. You can always use `--config config-backup.json` to start  
SD.Next back up with your previous settings, or undo the rename entirely if it did not help.

## 4. Use Debug Mode

If you're encountering errors of any kind, unexpected process terminations, or other issues, start up SD.Next with the `--debug`  
argument. This will allow you to see with greater detail what's going on, often exposing obvious fixes or indicating what the  
source of the errors are. In general we advise with running `--debug` all the time, but some users find it annoying to see it  
updating so often.

## 5. Safe Mode

Unless the issues you are having are directly involving an extension, it can be helpful to take all non-essential extensions  
out of the equation (disabling them) for troubleshooting purposes. Therefore we advise starting up with the `--safe` argument  
to see if any non-essential extensions are causing the issue at hand.
