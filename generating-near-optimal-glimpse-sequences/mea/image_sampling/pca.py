from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import torch
import pickle
import os
import time
from math import ceil
from progress.bar import Bar

from ..config import IMG_DIM, ATT_DIM, \
    FAKE_CELEBHQ_PCA_PATH, PCA_COMPONENTS, \
    FIT_PCA_BATCH_SIZE, N_DATA_POINTS_PCA
from .fake_loader import get_fake_loader

WEIGHTS_PATH = os.path.join(FAKE_CELEBHQ_PCA_PATH, "weights.p")
COMPONENTS_PATH = os.path.join(FAKE_CELEBHQ_PCA_PATH, "components.p")


def fit_pca():
    pca = IncrementalPCA(n_components=PCA_COMPONENTS)
    loader = get_fake_loader(FIT_PCA_BATCH_SIZE,
                             optional_flip=False)
    N_BATCHES = min(len(loader), ceil(N_DATA_POINTS_PCA/FIT_PCA_BATCH_SIZE))
    bar = Bar('Fitting', max=N_BATCHES)
    start = time.time()
    for i, batch in enumerate(loader):
        print(i)
        if i >= N_BATCHES:
            break
        X = batch.view(FIT_PCA_BATCH_SIZE, -1)
        pca.partial_fit(X)
        bar.suffix = f"({(i+1)*FIT_PCA_BATCH_SIZE}/{N_BATCHES*FIT_PCA_BATCH_SIZE}) | Time: {time.time()-start:.1f}s | "
        bar.next()
    components = pca.mean_, pca.components_
    pickle.dump(components, open(WEIGHTS_PATH, 'wb'))
    bar.finish()


def make_latents():
    pca = PCA()
    pca.mean_, pca.components_ = pickle.load(open(WEIGHTS_PATH, 'rb'))
    loader = get_fake_loader(FIT_PCA_BATCH_SIZE,
                             optional_flip=False)
    bar = Bar('Generating components', max=len(loader))
    start = time.time()
    for i, batch in enumerate(loader):
        print(i)
        latents = pca.transform(
            batch.reshape(FIT_PCA_BATCH_SIZE, 3*IMG_DIM**2)
        )
        if i == 0:
            all_latents = latents
        else:
            all_latents = np.concatenate((all_latents, latents), axis=0)
        # save progress after every batch
        pickle.dump(all_latents, open(COMPONENTS_PATH, 'wb'))
        bar.suffix = f"({(i+1)*FIT_PCA_BATCH_SIZE}/{len(loader)*FIT_PCA_BATCH_SIZE}) | Time: {time.time()-start:.1f}s | "
        bar.next()
    bar.finish()


class RegionPCA():

    def __init__(self, cuda, test=False):
        self.mean, self.weights = pickle.load(open(WEIGHTS_PATH, 'rb'))
        self.comps = pickle.load(open(COMPONENTS_PATH, 'rb'))
        self.mean, self.weights, self.comps = map(
            lambda t: torch.tensor(t).type(torch.FloatTensor),
            [self.mean, self.weights, self.comps])
        if test:
            self.comps = self.comps[:10000]
        if cuda:
            self.mean = self.mean.cuda()
            self.weights = self.weights.cuda()
            self.comps = self.comps.cuda()

    def get_approx_regions(self, R, C, n_chunks, flipped=False, padding=0):
        """
        Returns attention patch at (R, C) with specified padding
        and flip. Generates it in n_chunks chunks to reduce
        memory requirements.
        """
        assert R >= 0 and R < IMG_DIM - ATT_DIM
        assert C >= 0 and C < IMG_DIM - ATT_DIM
        if flipped:
            C = IMG_DIM - ATT_DIM - C

        # get required portions of weights
        padding_top = min(padding, R)
        padding_bottom = min(padding, IMG_DIM-ATT_DIM-1-R)
        padding_left = min(padding, C)
        padding_right = min(padding, IMG_DIM-ATT_DIM-1-C)
        R = R - padding_top
        C = C - padding_left
        EXTRACT_DIM_R = ATT_DIM + padding_top + padding_bottom
        EXTRACT_DIM_C = ATT_DIM + padding_left + padding_right
        region_indices = np.array([[[k*IMG_DIM**2+r*IMG_DIM+c
                                     for c in range(C, C+EXTRACT_DIM_C)]
                                    for r in range(R, R+EXTRACT_DIM_R)]
                                   for k in range(3)]).flatten()
        mean_ = self.mean[region_indices]
        components_ = self.weights[:, region_indices]

        # return generator of reconstructed regions for chunks of images
        return ((mean_ + torch.mm(chunk, components_)).view(-1, 3,
                                                            EXTRACT_DIM_R,
                                                            EXTRACT_DIM_C)
                for chunk in torch.chunk(self.comps, n_chunks))

    def get_latents(self, images):
        images = images.flatten(start_dim=1)
        images = images - self.mean
        return torch.mm(images, self.weights.transpose(0, 1))

    def get_full_images(self, latents):
        return (self.mean + torch.mm(latents, self.weights)).view(-1, 3, IMG_DIM, IMG_DIM)

    def reconstruct(self, images):
        return self.get_full_images(self.get_latents(images))
