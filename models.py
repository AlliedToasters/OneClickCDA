import torch
import torch.nn as nn

from processing import make_batch, get_image
import pandas as pd

import pickle
from processing import scale_data, tensorize, scale_outputs, batchify

from processing import get_scale, extract_proposal, scale_data, unscale_outputs, tensorize, decode_output
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class Predictor(object):
    """Does crater predictions."""
    def __init__(self, model):
        self.model = model

    def predict(self, prop_center, r, source, images):
        """
        Given a proposal (prop_center, r) and an image (images),
        applies the model and decodes its output.
        """
        img = images[source]
        scale = get_scale(r)
        X = extract_proposal(img, prop_center, scale)
        self.X = X
        X = np.expand_dims(X, axis=0)
        X = scale_data(X)
        X = tensorize(X)
        with torch.no_grad():
            Yhat = self.model(X)
        Yhat = Yhat.data.numpy()
        Yhat = unscale_outputs(Yhat)[0, :]
        x, y, r = decode_output(Yhat, prop_center, scale)
        return x, y, r

    def plot_prediction(self, prop_center, r, source, images):
        scale = get_scale(r)
        xhat, yhat, r = self.predict(prop_center, r, source, images)
        img = images[source]
        x_ = xhat - prop_center[0] + scale//2
        y_ = yhat - prop_center[1] + scale//2
        circle = matplotlib.patches.Circle((x_, y_), radius=r, fill=False)
        fig, ax = plt.subplots()
        ax.imshow(self.X)
        ax.add_artist(circle)
        plt.show()

class CraterModel(nn.Module):
    def __init__(self):
        super(CraterModel, self).__init__()
        #surface feature layers
        self.con1 = nn.Conv2d(3, 16, 3)
        self.con2 = nn.Conv2d(16, 32, 3, stride=2)
        self.con3 = nn.Conv2d(32, 32, 3, stride=2)

        #dimension reduction layers
        self.con4 = nn.Conv2d(32, 32, 3, stride=2)
        self.con5 = nn.Conv2d(32, 32, 3, stride=2)
        self.con6 = nn.Conv2d(32, 32, 3, stride=2)

        #size-correction layers (map features into shared space)
        self.fc32 = nn.Linear(1152, 1024)
        self.fc64 = nn.Linear(1152, 1024)
        self.fc128 = nn.Linear(1152, 1024)
        self.fc256 = nn.Linear(1152, 1024)

        #regression layers
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        #output layers
        self.circle_reg = nn.Linear(1024, 3)
        self.points_reg = nn.Linear(1024, 64)

        #nonlinearity
        self.activation = nn.ReLU()

    def surface_fts(self, x):
        x = self.con1(x)
        x = self.activation(x)
        x = self.con2(x)
        x = self.activation(x)
        x = self.con3(x)
        x = self.activation(x)
        return x

    def latent_fts(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def process_inputs(self, x):
        batch_dim = x.shape[0]
        img_dim = x.shape[-1]
        x = self.surface_fts(x)
        if img_dim == 32:
            x = x.reshape(batch_dim, -1)
            x = self.fc32(x)
            x = self.activation(x)
            x = self.latent_fts(x)
            return x
        x = self.con4(x)
        if img_dim == 64:
            x = x.reshape(batch_dim, -1)
            x = self.fc64(x)
            x = self.activation(x)
            x = self.latent_fts(x)
            return x
        x = self.con5(x)
        if img_dim == 128:
            x = x.reshape(batch_dim, -1)
            x = self.fc128(x)
            x = self.activation(x)
            x = self.latent_fts(x)
            return x
        x = self.con6(x)
        if img_dim == 256:
            x = x.reshape(batch_dim, -1)
            x = self.fc256(x)
            x = self.activation(x)
            x = self.latent_fts(x)
            return x
        else:
            msg = 'Invalid input image dimension: ' + str(img_dim)
            msg += '\nplease choose one of: 32, 64, 128, 256'
            raise Exception(msg)

    def regress_circle(self, x):
        fts = self.process_inputs(x)
        return self.circle_reg(fts)

    def regress_points(self, x):
        fts = self.process_inputs(x)
        return self.points_reg(x)

    def forward(self, x):
        return self.regress_circle(x)
