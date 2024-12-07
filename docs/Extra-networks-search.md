# Extra Networks Search

Search input for the extra networks accepts additional search syntax.
This allows to do more complicated searches, in addition to the searches which were previously available.

## Search Syntax

In a normal search, there are 3 special symbols which can be used to modify the search:

* `|` - *Or*, it allows to search for one of multiple terms at once (or exclude multiple terms, if prefixed with `-`)
* `-` - *Exclude*, terms or phrases prefixed with this symbol will be excluded from the search
* `&` - *And*, require all terms to be present (or not present, if prefixed with `-`) in the search results

These symbols are parsed in the order of `|`, `&`, `-`. This means that `a|b&c` is equivalent to "*a* or [*b* and *c*]" and `a&b|c` is equivalent to "[*a* and *b*] or *c*".
It is not possible to modify the order of operations with parentheses or other symbols.

Whitespace (this means spaces, tabs, newlines, etc.) is trimmed from individual terms.
This means that `a |    b` is equivalent to `a|b`.
Also `a|-b` is equivalent to `a| -   b`.

Search is case-insensitive, this means that `a` is equivalent to `A`.

However this concept might be clearer with a few examples based on a list of networks.

### Examples

Assume the following folder structure for your networks:

```text
├── SDXL/
│   ├── Clothing/
│   │   ├── Elegant Gown.safetensor
│   │   ├── Vintage Suit.safetensor
│   │   ├── Summer Dress.safetensor
│   │   └── Sports Attire.safetensor
│   ├── Concept/
│   │   ├── Futuristic Cityscape.safetensor
│   │   ├── Underwater World.safetensor
│   │   ├── Space Exploration.safetensor
│   │   └── Cyberpunk Street.safetensor
│   └── Person/
│       ├── Celebrity Portrait.safetensor
│       ├── Historical Figure.safetensor
│       ├── Fictional Character.safetensor
│       └── Movie Star.safetensor
└── SD1.5/
    ├── Clothing/
    │   ├── Casual Wear.safetensor
    │   ├── Casual Winter-Wear.safetensor
    │   ├── Winter Coat.safetensor
    │   ├── Evening Gown.safetensor
    │   └── Vintage Suit.safetensor
    ├── Concept/
    │   ├── Alien Landscape.safetensor
    │   ├── Medieval Castle.safetensor
    │   ├── Cyberpunk Street.safetensor
    │   └── Futuristic Cityscape.safetensor
    └── Person/
        ├── Scientist Portrait.safetensor
        ├── Movie Star.safetensor
        ├── Mythological Hero.safetensor
        └── Historical Figure.safetensor
```

Below are some example queries, which are explained in detail afterwards.

* `Vintage Suit` - Search for everything which contains the phrase `Vintage Suit`, meaning in this case `Vintage Suit.safetensor` in `SDXL` and `Vintage Suit.safetensor` in `SD1.5`.
* `clothing` - Search for everything which contains the word `clothing`, meaning in this case all networks in the `Clothing` folders.
* `clothing & SDXL` - Search for everything which contains the word `clothing` and the word `SDXL`, meaning in this case all networks in the `Clothing` folder of the `SDXL` folder.
* `-SD1.5` - Search for everything which does not contain the word `SD1.5`, meaning in this case all networks in the `SDXL` folder.
* `gown | dress` - Search for everything which contains the word `gown` or the word `dress`, meaning in this case `Evening Gown.safetensor` in `SD1.5` and `Elegant Gown.safetensor, Summer Dress.safetensor` in `SDXL`.
* `SDXL & gown | SDXL & dress` - Search for everything which contains the word `gown` or the word `dress` and the word `SDXL`, meaning in this just case `Elegant Gown.safetensor, Summer Dress.safetensor` in `SDXL`.
* `clothing | - Casual Wear` - Search for everything which contains the word `clothing`, as long as it does not contain the phrase `Casual Wear`, meaning in this case `Casual Winter-Wear.safetensor, Elegant Gown.safetensor, Vintage Suit.safetensor, Summer Dress.safetensor, Sports Attire.safetensor` in `SDXL` and `Evening Gown.safetensor, Vintage Suit.safetensor, Winter Coat` in `SD1.5`.
* `clothing | - Casual & Wear` - Search for everything which contains the word `clothing`, as long as it does not contain **both** the word `casual` and the word `wear`, meaning in this case `Elegant Gown.safetensor, Vintage Suit.safetensor, Summer Dress.safetensor, Sports Attire.safetensor` in `SDXL` and `Evening Gown.safetensor, Vintage Suit.safetensor, Winter Coat.safetensor` in `SD1.5` *(Note that `Casual Winter-Wear.safetensor` is excluded from the search results)*.
* `clothing | -black | - gray | -white` - Search for everything which contains the word `clothing`, as long as it does not contain the words `black`, `gray` or `white`.
* `SDXL | -Turbo | -LCM` - Search for everything which contains the word SDXL, as long as it does not contain the words `Turbo` or `LCM`. Can be used to filter different types of networks.
* `SDXL & Turbo | SDXL & LCM` - The inverse of the previous search. Search for everything which contains the word `Turbo` or `LCM` and the word `SDXL`.

### Advanced usage

It is also possible to do a RegEx search instead of a normal search.
To do so, the search term has to be prefixed with `r#`.
Everything after that will be interpreted as a RegEx search.


This means that `r#.*` will match everything, `r#^$` will match nothing and `r#a` will match everything which contains the letter `a`.
Please watch out that valid RegEx syntax is used, otherwise the search will fail.
In addition to that, the RegEx search is also slower than a normal search, so it is recommended to only use it if necessary.
RegEx searches are case-insensitive too.

One more thing to watch out for, is file paths. For the normal search the Windows `\` is replaced with a `/` to make it easier to search for file paths. This is not the case for the RegEx search. So you could use for example `[\/\\]` to match both `\` and `/` in a RegEx search.

Considering the RegEx search, the field which is searched is structured in the following way:

```JS
`filename: ${filename}|name: ${name}|tags: ${tags}`
```

As of writing this, the fields represent the following:

* `name` - The relative path to the network, not including filetype (also shows when hovering the network card for a longer time)
* `filename` - The absolute path to the network. (This does not resolve symlinks, so the path shown is the path to the symlink if one is used, not the path to the file the symlink points to)
* `tags` - The tags the network has, separated by `|`. Not all networks necessarily have tags, and they can be either keywords used when prompting, or the keyword to trigger the network specifically.


This allows to exclude (or include) certain strings only in specific fields.
For example a negative lookahead can be used to exclude certain tags from the search, while still searching for them in the title.

An example RegEx would be the following for this use case:

```RegEx
r#^(?!.*\|tags:.*(1girl|woman|girl)).*$
```

This search excludes every network which has tags which contain either `1girl`, `woman` or `girl`.

Another usecase would be searching in just the filename, but not in the entire path.
For example requiring the network to contain `SDXL`, with it not mattering whether it is in a folder called `SDXL` or not:

```RegEx
r#^.*\|name:[^|]*([^\/\\]*SDXL[^\/\\]*)$(?<![\/\\]).*$
```

As this is already advanced usage, the assumption is made that the user knows how to use RegEx, so no further examples or explanations of the provided RegEx are given here.
Tools like <https://regex101.com/> can be used to create, test, understand and debug a regular expression.
