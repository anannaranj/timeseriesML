# Project Workflow

This file isn't a part of the project, the best way to describe it is it's just some sort of diary

I used neovim as my editor, and ran the code directly in the terminal. With the power of vim keybindings and GNU/Linux I was able to have a little speed boost compared to programming in an IDE (e.g. VSCode).
For neovim I just used ![kickstart.nvim](https://github.com/nvim-lua/kickstart.nvim) configuration as I'm lazy to create my own config. Even with bare edits to the kickstart config you get a pretty much solid experience!
For the commands I used to run python, and with the power of the terminal you can make usual tasks really easy, Here is how:

I created this function in my ![fish](https://fishshell.com/) config:
```
function py
    python -i "$argv"
end
```
this let's me run the file like `py main.py` and after it is finished it will put me into the interactive python shell, was really useful for debugging and performing small queries on the dataframes.
Yes, I could've used `python -i main.py`, but come on we love aliases. :D

Of course I made a python environment so I can install packages just for this project, and freeze the requirements.
