# OneClickCDA

This is a project to increase the productivity of crater annotators. Using convolutional neural networks, a model regresses the exact location of crater given a rough proposal.

<iframe width="560" height="315" src="https://www.youtube.com/embed/a4d1GfKvgUs" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

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

(Hint: install libraries using pip:<br>
`pip install -r requirements.txt`<br>
using requirements.txt file found in this repo)

Running the prototype program:
clone this repository<br>
`git clone https://github.com/AlliedToasters/OneClickCDA.git`<br>

from OneClickCDA directory, run:<br>

`python deploy.py`<br>
-OR-<br>
`python3 deploy.py`<br>
(depends on your system, but make sure you're using python 3.6)<br>

Type in the path to an image (should be reasonably small, 2,000x2,000 px or less)<br>

Matplotlib should open a window on your system.<br>

Controls:<br>
`right click - switch proposal size`<br>
`left click - propose crater center`<br>
`u - undo previous detection`<br>

Close the window, and the command line will ask you where to store your results. Enter a path to the desired filename.
