#Script to contain production prototype for model.
import pandas as pd
from models import CraterModel, Predictor
from processing import get_image
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class DetectionManager(object):
    """
    Manages detection events.

    Keeps a running record of detections
    """
    def __init__(self, img, predictor_path ='./models/predictor1.pkl'):
        with open(predictor_path, 'rb') as f:
            self.predictor = pickle.load(f)
        self.results = None
        self.images = dict()
        self.images['dummy'] = img
        self.current_r = 0

    def add_results(self, x, y, r, source):
        if self.results is None:
            self.results = pd.DataFrame(columns=['x', 'y', 'r', 'source'])
        cols = self.results.columns
        row = pd.DataFrame(columns=cols, index=[len(self.results)])
        row['x'] = [round(x, 2)]
        row['y'] = [round(y, 2)]
        row['r'] = [round(r, 2)]
        row['source'] = source
        self.results = pd.concat([self.results, row], axis=0)

    def do_detection(self, x, y, r):
        prop_center = (int(x), int(y))
        source = 'dummy'
        x, y, r = self.predictor.predict(prop_center, r, source, self.images)
        self.add_results(x, y, r, source)

    def write_csv(self, path, incl_source=False):
        if incl_source:
            self.results.to_csv(path, index=False)
        else:
            self.results[['x', 'y', 'r']].to_csv(path, index=False)

    def undo_result(self):
        if self.results is None:
            return
        elif len(self.results) == 0:
            return
        else:
            self.results = self.results.iloc[:-1]
            return

manager = None

def onclick(event):
    global manager
    rs = [7, 15, 31, 50]
    button = event.button.value
    if str(button) == '3':
        manager.current_r += 1
        if manager.current_r == 4:
            manager.current_r = 0
        retitle(manager)
    elif str(button) == '1':
        x = event.xdata
        y = event.ydata
        r = rs[manager.current_r]
        manager.do_detection(x, y, r)
        redraw(manager)
    else:
        print(button)

def press(event):
    letter = event.key
    if letter == 'u':
        global manager
        manager.undo_result()
        plt.cla()
        ax = plt.gca()
        ax.imshow(manager.images['dummy'][:, :, 0], cmap='Greys_r')
        redraw(manager)
        retitle(manager)

def retitle(manager):
    rs = [8, 16, 32, 100]
    ax = plt.gca()
    if manager.current_r == 0:
        range = '0-8 pixels'
    elif manager.current_r == 3:
        range = '32-100 pixels'
    else:
        range = f'{rs[manager.current_r-1]}-{rs[manager.current_r]} pixels'
    tit = 'current radius range: ' + range
    ax.set_title(tit)
    plt.draw()

def redraw(manager):
    ax = plt.gca()
    results = manager.results
    if results is not None:
        for i, row in results.iterrows():
            x = row.x
            y = row.y
            r = row.r
            circle = Circle((x, y), r, fill=False, color='purple')
            ax.add_artist(circle)
    plt.draw()

def detection_loop(manager):
    fig, ax = plot_results(manager)
    retitle(manager)
    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_press = fig.canvas.mpl_connect('key_press_event', press)
    plt.show()

def plot_results(manager):
    img = manager.images['dummy']
    results = manager.results
    if results is None:
        results = pd.DataFrame()
    fig, ax = plt.subplots()
    ax.imshow(img[:, :, 0], cmap='Greys_r')
    for i, row in results.iterrows():
        x = row.x
        y = row.y
        r = row.r
        circle = Circle((x, y), r, fill=False)
        ax.add_artist(circle)
    return fig, ax

if __name__ in '__main__':
    results = None
    source = input('enter image path: ')
    img = get_image(source, absolute=True)
    manager = DetectionManager(img)
    detection_loop(manager)
    results_path = input('Path to save results? ')
    manager.write_csv(results_path)
