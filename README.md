# OneClickCDA

This is a project to increase the productivity of crater annotators. Using convolutional neural networks, a model regresses the exact location of crater given a rough proposal.

*This project is still in the early prototype phase*

To give it a try, here are the requirements:

Python version: `3.6`

Libraries:
```
torch
matplotlib
pandas
numpy
opencv-python
```

(Hint: install libraries using pip:\n
`pip install -r requirements.txt`\n
using requirements.txt file found in this repo)

Running the prototype program:
clone this repository
`git clone https://github.com/AlliedToasters/OneClickCDA.git`\n

from OneClickCDA directory, run:\n

`python deploy.py`\n
-OR-\n
`python3 deploy.py`\n
(depends on your system, but make sure you're using python 3.6)\n

Type in the path to an image (should be reasonably small, 2,000x2,000 px)

Matplotlib should open a window on your system.

Controls:
`right click - switch proposal size`
`left click - propose crater center`
`u - undo previous detection`

Close the window, and the command line will ask you where to store your results. Enter a path to the desired filename.
