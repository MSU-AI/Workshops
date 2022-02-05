# Workshops
This repository contains workshops previously hosted by the 
MSU AI club.

Each workshop is contained within their own folders.
More info on each individual workshop can be found
within their respective folders.

# Python
All of the workshops utilise the [python programming language](https://www.python.org) in some way.
Because of this, it is necessary for python to be installed before working with these workshops.

Windows users can download python [here](https://www.python.org/downloads/windows/),
and MacOS users can download python [here](https://www.python.org/downloads/macos/).
You probably want the latest version.

Linux users should download python from their distribution's package manager.
You can also find the source code for python on the website linked above
if you wish to build from source.

# Jupyter Notebooks
Pretty much all of the workshops are [Jupyter Notebooks](https://jupyter.org).
Notebook files(extension .ipynb) require software for the contents to display correctly.

The recommended editor for working with notebook files is [vscode](https://code.visualstudio.com),
which should have notebook extensions installed by default.
You should also install the necessary python extensions to enable 
rich python editing. You can find out how to do this [here](https://code.visualstudio.com/docs/languages/python).

# Other Dependencies
Most notebooks have dependencies that are required for it to work correctly.
Each workshop folder has a file called `requirements.txt` which contains the
names of required dependencies.

You can install these dependencies like so:

`pip install -r requirements.txt`

Note that your terminal must be in the directory of the workshop you wish to work with,
and the `pip` command will not become available until you install python. 

It is recommended to use [virtual environments](https://docs.python.org/3/tutorial/venv.html) to ensure that these dependencies do not clutter your python install.
This step is optional, but highly recommended!

# Index
- airline-sentiment - Airline Tweet sentiment analysis
- opencv-into - Intro to OpenCV
