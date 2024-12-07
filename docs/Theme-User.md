# User Themes

Creating custom themes can be done with minimal knowledge of `CSS` as each theme is a *single CSS file*

!!! tip

    While you're modifying a theme, its changes will be visible immediately on page refresh, no need for server restart  
    However, make sure that you clear browser cache between edits  
    Easiest approach is to open browser inspector window (`F12` in Chrome) and select **disable cache** in **network** tab  
    You can also experiment with live edits in browser inspector and only copy them to theme once you're satisfied with changes  

## Standard UI

- Theme is a CSS file in `/javascript` folder
- Copy existing theme file, for example `black-teal.css` or `light-teal.css` into new file, for example `my-theme.css`
- Edit `my-theme.css` to change colors, fonts, sizes, borders, paddings, margings, etc.
- Theme will be selectable in UI after server restart

## Modern UI

- Theme is a CSS file in `/extensions-builtin/sdnext-modernui/themes` folder
- Copy existing theme file, for example `default.css` into new file, for example `my-theme.css`
- Edit `my-theme.css` to change colors, fonts, sizes, borders, paddings, margings, etc.
- Theme will be selectable in UI after server restart

### Contributing

Once you're happy with your theme, you can share it with the community by submitting a pull request that includes your CSS file!

- For Standard UI, create PR in [SDNext repo](https://github.com/vladmandic/automatic/pulls)
- For Modern UI, create PR in [ModernUI repo](https://github.com/BinaryQuantumSoul/sdnext-modernui/pulls)

But...If you're not comfortable with that, you can always share your theme in [Discussions](https://github.com/vladmandic/automatic/discussions)
