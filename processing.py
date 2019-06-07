import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from cv2 import equalizeHist, createCLAHE

def standardize_image(img):
    amu = 99.5
    asd = 33.2
    mu = img.mean()
    sd = img.std()
    dmu = mu-amu
    dsd = sd/asd
    new_img = img.copy().astype(float)
    new_img -= dmu
    olmean = new_img.mean()
    new_img -= olmean
    new_img /= dsd
    new_img += olmean
    new_img = np.where(new_img>255, 255, new_img)
    new_img = np.where(new_img<0, 0, new_img)
    new_img = new_img.astype(np.uint8)
    return new_img

def make_channels(input_image):
    """Converts image to three channels:
    1: unaltered image
    2: Histogram equalized image
    3: CLAHE image
    """
    ch1 = np.array(input_image)
    ch2 = equalizeHist(ch1)
    clahe = createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ch3 = clahe.apply(ch1)
    image = [ch1, ch2, ch3]
    image = [np.expand_dims(x, axis=-1) for x in image]
    image = np.concatenate(image, axis=-1)
    return image

def get_image(crater, clahe=True, absolute=False):
    """Loads an image to memory, returns array"""
    if absolute:
        path = crater
    else:
        name = crater.split('.')[0]
        path = './images/{}.png'.format(name)
    img = cv2.imread(path)
    if img is None:
        raise Exception('Image file not found.')
    img = standardize_image(img)
    if clahe:
        img = make_channels(img[:, :, 0])
    return img

def get_grid(row, out_dim=32, r_ratio=4):
    ctr = (row.x, row.y)
    startx = ctr[0] - (row.r * r_ratio)
    stopx = ctr[0] + (row.r * r_ratio)
    starty = ctr[1] - (row.r * r_ratio)
    stopy = ctr[1] + (row.r * r_ratio)
    length = stopx - startx
    step_size = length/(out_dim-1)
    map_x = np.array([np.arange(startx, stopx+1e-5, step_size) for x in range(out_dim)], dtype=np.float32)
    map_y = np.array([np.arange(starty, stopy+1e-5, step_size) for x in range(out_dim)], dtype=np.float32)
    map_y = map_y.T
    return map_x, map_y

def get_aligned(images, row):
    """Returns aligned crater"""
    image = images[row.source]
    map_x, map_y = get_grid(row)
    mapped_img = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return mapped_img

def extract_crater(row, images):
    """Extracts crater from annotation and image"""
    aln = get_aligned(images, row)
    return aln

def plot_example(mapped_img):
    dim = mapped_img.shape[0]
    plt.imshow(mapped_img)
    plt.axvline(x=dim/4, c='r', linestyle='dotted')
    plt.axvline(x=(dim*3)/4, c='r', linestyle='dotted')
    plt.axhline(y=dim/4, c='r', linestyle='dotted')
    plt.axhline(y=(dim*3)/4, c='r', linestyle='dotted')
    sct_dim = (dim-1)/2
    plt.scatter(x=[sct_dim], y=[sct_dim], c='r')
    plt.show()

def create_labeled_example(row, images, dx, dy, dr):
    """Creates a labeled example"""
    label = np.array([dx, dy, dr]).reshape(1, -1)
    scl = row.r
    modified_row = row.copy()
    modified_row.r = row.r + dr * scl
    modified_row.x = row.x + dx * scl
    modified_row.y = row.y + dy * scl

    extr = extract_crater(modified_row, images)
    extr = np.expand_dims(extr, axis=0)
    return extr, label

def get_images(csv_path='./annotations/train.csv', img_dir='./images/'):
    df = pd.read_csv(csv_path)
    tiles = list(df.source.unique())
    images = dict()
    for tile in tiles:
        img = get_image(tile)
        images[tile] = img
    return df, images

def tensorize(X):
    X = torch.tensor(X)
    X = X.permute((0, 3, 1, 2)).float()
    return X

def imcrop(img, bbox):
    """Crops an image with padding. Given

    img: an array
    bbox: a bounding box 4-tuple (x1, y1, x2, y2)

    returns the desired bounding box,
    replicating border pixels when necessary
    """
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                            -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

def extract_proposal(img, center, scale):
    """
    Given a crater proposal, this function extracts
    the important pixels in a consistent manner.
    img: an array
    center: (x, y) coordinates (integers) for center proposal
    scale: one of [32, 64, 128, 256] (very rough size of crater)
        follows the scheme:
        0<r<8 --> scale=32
        8<r<16 --> scale=64
        16<r<32 --> scale=128
        32<r --> scale=256
        (craters bigger than r=64 not supported)
    """
    permitted_scales = [32, 64, 128, 256]
    if scale not in permitted_scales:
        msg = f"scale {scale} not permitted. Please use one of: "
        msg += str(permitted_scales)
        raise Exception(msg)
    else:
        scl = scale//2
    xmin = center[0] - (scl-1)
    ymin = center[1] - (scl-1)
    xmax = center[0] + (scl+1)
    ymax = center[1] + (scl+1)
    return imcrop(img, (xmin, ymin, xmax, ymax))

def create_labeled_pair(img, gt_center, prop_center, gt_radius, scale):
    """
    Given a crater proposal and ground truth label,
    this function creates a labeled pair.
    Returns X, Y, where X is an image, and Y is a set of ground truths.
    img: an array
    gt_center: the known ground-truth center point (x, y) (floats)
    prop_center: the crater proposal center (x, y) (ints)
    gt_radius: the known ground-truth crater radius in pixels (float)
    scale: one of [32, 64, 128, 256] (very rough size of crater)
        follows the scheme:
        0<r<8 --> scale=32
        8<r<16 --> scale=64
        16<r<32 --> scale=128
        32<r --> scale=256
        (craters bigger than r=64 not supported)
    """
    permitted_scales = [32, 64, 128, 256]
    if scale not in permitted_scales:
        msg = f"scale {scale} not permitted. Please use one of: "
        msg += str(permitted_scales)
        raise Exception(msg)
    scale_factor = scale//32
    x_offset = (gt_center[0] - prop_center[0])/scale_factor
    y_offset = (gt_center[1] - prop_center[1])/scale_factor
    r_scaled = gt_radius/scale
    Y = (x_offset, y_offset, r_scaled)
    X = extract_proposal(img, prop_center, scale)
    return X, Y

def get_scale(r):
    """Given a radius, returns the appropriate scale."""
    try:
        assert r>0
    except AssertionError:
        raise Exception(msg)
    if r < 8:
        return 32
    elif r < 16:
        return 64
    elif r < 32:
        return 128
    elif r < 128:
        return 256
    else:
        return -1

def generate_proposal(row, sigma=1.0):
    """
    Given a ground truth crater annotation,
    adds noise and generates proposal.
    Returns:
        prop_center: (x, y) locations (ints) of approximate center
        scale: scale of proposal.
    """
    #scale back noise on radius - not important
    s = sigma/10
    rnoise = (np.random.randn()*s) + 1
    r = row.r * rnoise
    scale = get_scale(r)
    scale_factor = scale//32
    xnoise = np.random.randn()*sigma*scale_factor
    x = round(row.x + xnoise)
    ynoise = np.random.randn()*sigma*scale_factor
    y = round(row.y + ynoise)
    prop_center = (x, y)
    return prop_center, scale

def decode_output(Yhat, prop_center, scale):
    """Transforms model output into original image space"""
    scale_factor = scale//32
    xhat = (Yhat[0]*scale_factor) + prop_center[0]
    yhat = (Yhat[1]*scale_factor) + prop_center[1]
    r = Yhat[2]*scale
    return xhat, yhat, r

def generate_data(df, images, batch_size=32, scale=32, sigma=1.0):
    bounds = get_bounds(scale)
    cond1 = df.r>bounds[0]
    cond2 = df.r<bounds[1]
    cond = cond1 & cond2
    slc = df[cond]
    X = None
    Y = None
    for i, row in slc.sample(frac=5, replace=True).iterrows():
        prop_center, scl = generate_proposal(row, sigma=1.0)
        if scl != scale:
            continue
        img = images[row.source]
        gt_center = (row.x, row.y)
        gt_radius = row.r
        X_, Y_ = create_labeled_pair(img, gt_center, prop_center, gt_radius, scl)
        X_ = np.expand_dims(np.array(X_), axis=0)
        Y_ = np.expand_dims(np.array(Y_), axis=0)
        if X is None:
            X = X_
            Y = Y_
        else:
            X = np.concatenate([X, X_], axis=0)
            Y = np.concatenate([Y, Y_], axis=0)
        if len(X) == batch_size:
            break
    return X, Y

def batchify(X, Y):
    X = scale_data(X)
    X = tensorize(X)
    Y = scale_outputs(Y)
    Y = torch.tensor(Y).float()
    return X, Y

def make_batch(df, images, batch_size=32, scale=32, sigma=1.0):
    X, Y = generate_data(df=df, images=images, batch_size=batch_size, scale=scale, sigma=sigma)
    X, Y = batchify(X, Y)
    return X, Y

def scale_data(X):
    mu0 = 91.2
    sd0 = 22.4
    mu1 = 107.9
    sd1 = 66.8
    mu2 = 96.3
    sd2 = 26.0
    X = X.copy().astype(float)
    X[:, :, :, 0] -= mu0
    X[:, :, :, 1] -= mu1
    X[:, :, :, 2] -= mu2
    X[:, :, :, 0] /= sd0
    X[:, :, :, 1] /= sd1
    X[:, :, :, 2] /= sd2
    return X

def scale_outputs(Y):
    mu0 = 0
    sd0 = 1.0
    mu1 = 0
    sd1 = 1.0
    mu2 = 0.157
    sd2 = 0.05
    Y = Y.copy().astype(float)
    Y[:, 0] -= mu0
    Y[:, 1] -= mu1
    Y[:, 2] -= mu2
    Y[:, 0] /= sd0
    Y[:, 1] /= sd1
    Y[:, 2] /= sd2
    return Y

def unscale_outputs(Y):
    mu0 = 0
    sd0 = 1.0
    mu1 = 0
    sd1 = 1.0
    mu2 = 0.157
    sd2 = 0.05
    Y = Y.copy().astype(float)
    Y[:, 0] *= sd0
    Y[:, 1] *= sd1
    Y[:, 2] *= sd2
    Y[:, 0] += mu0
    Y[:, 1] += mu1
    Y[:, 2] += mu2
    return Y

def get_bounds(scale):
    if scale==32:
        return 0, 12
    elif scale==64:
        return 6, 24
    elif scale==128:
        return 12, 40
    elif scale==256:
        return 25, 130
