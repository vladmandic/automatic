# MacOS and Python

[TL;DR: Installation Instructions](#installation-instructions)

The MacOS operating system requires Python, so it is installed by default.  However, if you are going to start using Python on your own, it is likely that you'll want to install new packages, perform package updates, and so on -- which can be a problem for MacOS -- it's important to let the operating system handle its own Python.

Most sources online will tell you to use Homebrew as a package manager for MacOS, so it is a natural conclusion to consider using Homebrew for Python -- and it's common to discover that you already have Python installed through Homebrew as a dependency for something else.  However, this isn't the correct solution either -- it's really the same problem as the MacOS problem, but with a coat of paint.

Homebrew's Python is there to support other packages.  Importantly, Homebrew deletes old versions of packages after 30 days.  So, if you are using, say, python 3.12 for random scripting and other tasks, but need version 3.10 for, say, SD.Next, Homebrew will delete 3.10, since it doesn't know that you need it for something (since you didn't install SD.Next via Homebrew).  

The solution is to use another way to manage the Python version(s) that you use on your own.  I use [asdf](https://asdf-vm.com/), which has a [Python plugin](https://github.com/asdf-community/asdf-python), but there are others if you prefer something else.

Sources / Further Reading: 
- <https://justinmayer.com/posts/homebrew-python-is-not-for-you/>
- <https://hackercodex.com/guide/python-development-environment-on-mac-osx/>
- <https://github.com/asdf-community/asdf-python>
- <https://asdf-vm.com/>
- <https://docs.brew.sh/Installation>

## Installation Instructions

1. If you haven't got Homebrew installed already:

```zsh
mkdir homebrew && curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip-components 1 -C homebrew

eval "$(homebrew/bin/brew shellenv)"
brew update --force --quiet
chmod -R go-w "$(brew --prefix)/share/zsh"
```

2. Install asdf and python build dependencies:

```zsh
brew install asdf openssl readline sqlite3 xz zlib
```

3. Add asdf to `.zshrc` to use it immediately and persistently:

```zsh
. $(brew --prefix asdf)/asdf.sh
echo -e "\n. $(brew --prefix asdf)/asdf.sh" >> ~/.zshrc
```

4. Add the python asdf plugin:

```zsh
asdf plugin add python

# for SD.Next
asdf install python 3.10.14

# you may want the latest version too; take note of which version is installed
asdf install python latest
```

5. Set the default global version of python. Since you will always want 3.10 for SD.Next, you will want to always specifically use that version.  You will probably want to use the command `python` in most contexts, and `python3.10` for version-specific uses.

```zsh
asdf global python 3.12.2 
# or whatever version you installed
```

6. Run SD.Next using python3.10:

```zsh
export PYTHON=$(which python3.10)
cd /path/to/SD.Next
./webui.sh --debug
```
