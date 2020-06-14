import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from pthflops import count_ops
from PIL import Image


def str2bool(v):
    return v.lower() in ('true', '1')

def denormalize(T, coords):
    return (0.5 * ((coords + 1.0) * T))


def bounding_box(x, y, size, color='w'):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class BatchScoreMeter(object):
    """
    Computes F1 etc. from batched versions of TP, FP etc.
    """
    def to_array(self, *args):
        args = [np.array(arg).reshape(self.TP.shape) for arg in args]
        return np.stack(args + [self.TP, self.FP, self.TN, self.FN], axis=-1)

    def load_from_array(self, array):
        losses = array[:, :-4]
        self.TP, self.FP, self.TN, self.FN = (array[:, i] for i in [-4, -3, -2, -1])
        return losses

    def to_csv(self, *args):
        array = self.to_array(*args)
        return "\n".join(','.join(map(str, row)) for row in array)+'\n'

    def load_from_csv(self, csv):
        csv = csv.rstrip('\n').split('\n')
        array = np.array([[float(f) for f in row.split(',')] for row in csv])
        return self.load_from_array(array)

    def num_total(self):
        return self.TP + self.FP + self.TN + self.FN

    def num_correct(self):
        return self.TP + self.TN

    def accuracy(self):
        return self.num_correct() / self.num_total()

    def balanced_accuracy(self):
        acc1 = self.TP / (self.TP + self.FN)
        acc2 = self.TN / (self.TN + self.FP)
        return (acc1 + acc2)/2

    def F1(self):
        recall = self.TP / (self.TP+self.FN)
        precision = self.TP / (self.TP+self.FP)
        f1_score = 2 * (precision*recall / (precision+recall))
        f1_score[np.isnan(f1_score)] = 0
        return f1_score

    def balanced_F1(self):
        return 0.5*(self.F1() + self.reverse().F1())

    def reverse(self):
        new = type(self)()
        new.TP = self.TN
        new.TN = self.TP
        new.FP = self.FN
        new.FN = self.FP
        return new

    def dataset_prop_positive(self):
        return (self.TP+self.FN)/self.num_total()

class ScoreMeter(BatchScoreMeter):
    """
    Takes in labels and predictions, computes
    and stores accuracy, F1 scores etc.
    """
    def __init__(self):
        self.TP = np.array([0])
        self.FP = np.array([0])
        self.TN = np.array([0])
        self.FN = np.array([0])

    def update(self, labs, pred):
        def arr(x): return np.array([x])
        self.TP += arr((labs * pred).sum())
        self.FP += arr(((1-labs) * pred).sum())
        self.FN += arr((labs * (1-pred)).sum())
        self.TN += arr(((1-labs) * (1-pred)).sum())


def resize_array(x, size):
    # 3D and 4D tensors allowed only
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    # 4D Tensor
    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            img = array2img(x[i])
            img = img.resize((size, size))
            img = np.asarray(img, dtype='float32')
            img = np.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = np.concatenate(res)
        res = np.expand_dims(res, axis=1)
        return res

    # 3D Tensor
    img = array2img(x)
    img = img.resize((size, size))
    res = np.asarray(img, dtype='float32')
    res = np.expand_dims(res, axis=0)
    res /= 255.0
    return res


def img2array(data_path, desired_size=None, expand=False, view=False):
    """
    Util function for loading RGB image into a numpy array.

    Returns array of shape (1, H, W, C).
    """
    img = Image.open(data_path)
    img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype='float32')
    if expand:
        x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def array2img(x):
    """
    Util function for converting anumpy array to a PIL img.

    Returns PIL RGB img.
    """
    x = np.asarray(x)
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype('uint8'), 'RGB')


def plot_images(images, gd_truth):

    images = images.squeeze()
    assert len(images) == len(gd_truth) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i], cmap="Greys_r")

        xlabel = "{}".format(gd_truth[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def prepare_dirs(config):
    for path in [config.ckpt_dir, config.logs_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def unnormalise_img(x):
    """
    undoes a normalisation with 0.5 mean and 0.5 std deviation
    """
    eps = 0.001
    mean = torch.Tensor((0.5,)).view(1, 1, 1)  # .repeat(1, dim, dim)
    std = torch.Tensor((0.5,)).view(1, 1, 1)  # .repeat(1, dim, dim)
    x = mean + x*(std*(1+eps))
    return x


def to_pil(img):
    """
    WARNING: the denormalisation stuff is hardcoded
    """
    grayscale = img.shape[0] == 1
    img = unnormalise_img(img)
    img = np.uint8(
            255*img.numpy()
    ).transpose(1, 2, 0)
    if grayscale:
        img = img[:, :, 0]
    return Image.fromarray(
        img,
        'L' if grayscale else 'RGB'
    )


class BestTracker():
    def __init__(self, attrs):
        """
        Enter dictionary of attribute name and comparator # (such that a # b
        is True when a better than b)
        """
        self.attrs = {attribute_name: [comparator, None] for attribute_name, comparator in attrs.items()}

    def compare(self, comparator, a, b):
        if type(comparator) == str:
            comparator = {'greater': lambda a, b: a>=b,
                          'lesser': lambda a, b: a<=b,
                          }[comparator]
        return comparator(a, b)

    def update(self, attribute_name, value):
        """
        Update an attribute with a new, possibly best, value. Returns True if
        this was indeed a new best value and otherwise returns False.
        """
        comparator, best_val = self.attrs[attribute_name]
        new_best = (best_val is None) or self.compare(comparator, value, best_val)
        if new_best:
            self.attrs[attribute_name][1] = value
        return new_best

def get_flops(*args):
    return count_ops(*args, print_readable=False, verbose=False)[0]

def get_performance(prediction_log_probs, y, metric):
    # B x C prediction_probs, B y, 'nll' or 'acc'
    # returns B-dimensional tensor
    if y is None:
        # not supervised - so take an expectation over predictions_probs
        y_probs = torch.softmax(prediction_log_probs, dim=-1)
        performance = 0
        for y_val in range(y_probs.shape[1]):
            y = y_val * torch.ones(y_probs.shape[0]).long()
            performance = performance + y_probs[:, y_val] * \
                get_performance(
                    prediction_log_probs,
                    y, metric)
        return performance

    if metric == 'nll':  # then, confusingly, we return positive ll
        return -F.nll_loss(prediction_log_probs,
                           y, reduction='none')
    elif metric == 'acc':
        actual_predictions = torch.argmax(prediction_log_probs, dim=-1)
        return (y == actual_predictions).float()
    else:
        raise Exception('Unrecognised early stopping type.')
