# Extensions Development

## Common Mistakes

- **Execution on import**:  
  There should be NO CODE that executes on import in any of the extension files (with exception of allowed trivial `install.py` code)  
  All code should be in functions/classes and executed on application callbacks  
  Failure to do so results in slow server startup (at best) or even server crashes (at worst)  
- **Keeping all Python code in `/scripts`**:  
  That folder should have a single entry point file and that file should import any other file that may reside anywhere else EXCEPT `/scripts` folder  
  Failure to do so results server loading and executing each and every file during startup and then extension itself performs imports again  
  If extension fails to do so results in slow server startup (at best) or even server crashes (at worst)  
- **Browser namespace collisions**:  
  All JS code is imported as-is and lives in an global browser namespace  
  Do not define functions like `log()` - use a unique prefix for all your functions and variables  
- **Incorrect usage of callbacks**:  
  For example, JS code can be executed `onUiLoaded` or `onUiUpdated`  
  First one is triggered once, second one is triggered hundreds of times during server startup  
  Choosing wrong callback to perform initialization work results in initializaton work being performed hundreds of times resulting in slow page loads  
  Typical problem is why is my browser page loading so slow or why is autolaunch not working? Because browser is asked to do the same things hundreds of times  
  This applies to both Python and JS code - always check if you're using correct callbacks  
- **Executing code when disabled**:  
  If extension is not enabled, it should not perform any work in callbacks it is registered for and always perform early-exit when work is not needed  
  This can result in pre/during/post generate delays for no apparent reasons  
  Typical problem would be ~0.5sec delay before generate start - which quickly adds up to overall time-to-generate regardless of how fast generate actually is  
- **Unsafe references**:  
  Extensions can access server variables either directly or as provided via callbacks, but content of those variables should be handled with care  
  For example, extension can access `image.info` property, but there is no guarantee that property will exist for all images - always use safe access methods like `get()` or `getattr()`  
  This also applies to server settings - extensions cannot assume settings never change, otherwise its not viable to ever improve underlying server at all  
- **Unsafe patching**:  
  Extensions can patch server code by providing overrides for some methods, but should do so with care  
  For example, extension can replace `forward` functions, but always consider there may be other extensions that do the same  
  So patching should never be done globally on in a way that breaks other extensions  
- **Running platform specific code**:
  Not everyone runs the same OS or compute backend and using platform or hardware specific code or relying on packages which are not cross-platform is just bad  
  For example, never use platform specific designators such as `torch.to('cuda')` or `torch.float16`, always use well-defined server variables such as `device.device` and `device.dtype`  
- **Assuming values**:
  Never assume values, always check a well-defined variable what is the current value and handle it accordingly
  For example, extension may not be installed in `/extensions` folder, it may be relocated to a different folder

Unfortunately, looking at top-20 most popular extensions, most of them are guilty of not just one or two, but majority of the above cases and this is not isolated only to extensions that are considered broken  
Always remember that by installing extension you give it full access to do anything and you're relying on extension author to perform all the work correctly and safely  

## Extension vs Script

- **Script** is a single module that implements a script
- **Script** object is inherited from `Script` class and implements several mandatory and any of the optional methods for well-defined callbacks
- **Extension** is a larger implementation that exists as its own folder with a well-defined folder structure
- **Extension** code can further define any number of **Scripts** or work on its own

## Extension Folder Structure

Note: any of the files are optional

- `/preload.py`  
  Loaded early during server startup to provide additional command line parameters  
  Extension should define a `preload(parser: argparser)` function to extend parser as needed  
  Preload should perform no other work or access any other modules
- `/install.py`  
  Loaded early during server startup to install any optional requirements  
  Must use well defined server functions such as `launch.run_pip` and access only modules explicitly marked as safe at the early stages of server startup  
  Do not do direct OS calls or perform any other work other than basic installation
- `/javascript/*.js`  
  All JS files in a folder are added as-is
  There should be no work done in JS scripts other than defining functions/variables and registering callbacks that will be executed at appropriate time
- `/style.css`  
  Style is added as-is  
- `/scripts/*.py`  
  This is intended as a main entry-point for extension and every file is loaded by the server during startup
  There should be no work done in Python scripts other than defining functions/variables and registering callbacks that will be executed at appropriate time
