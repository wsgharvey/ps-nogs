from os.path import join
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
from progress.bar import Bar
from math import ceil
import time
from glob import glob
import pickle
from PIL import Image
from mea.utils import downsample, get_observed_patch, allow_unbatched

import torch
from torchvision import transforms
import torch.nn.functional as F

from mea.config import FAKE_BIRDS_PATH, \
    BIRDS_PCA_COMPONENTS as PCA_COMPONENTS, \
    BIRDS_PCA_BATCH_SIZE as FIT_PCA_BATCH_SIZE, \
    BIRDS_N_DATA_POINTS_PCA as N_DATA_POINTS_PCA, \
    BIRD_IMG_DIM, BIRD_ATT_DIM, BIRDS_MEAN, \
    BIRDS_STD
from .fake_loader import get_fake_loader, BatchlessFakeLoader

FAKE_BIRDS_PCA_PATH = join(FAKE_BIRDS_PATH, "pca")
def GET_WEIGHTS_PATH(image_type):
    return join(FAKE_BIRDS_PCA_PATH,
                f"{image_type}_weights.p")
def GET_COMPONENTS_PATH(image_type, N0, N):
    return join(FAKE_BIRDS_PCA_PATH,
                f"{image_type}_components_{N0}-{N0+N}.p")

def fit_pca(image_type, N):
    print("fitting")
    pca = IncrementalPCA(n_components=PCA_COMPONENTS)
    loader = get_fake_loader(image_type,
                             FIT_PCA_BATCH_SIZE,
                             optional_flip=False,
                             N=N)
    N_BATCHES = min(len(loader), ceil(N_DATA_POINTS_PCA/FIT_PCA_BATCH_SIZE))
    bar = Bar('Fitting', max=N_BATCHES)
    start = time.time()
    for i, batch in zip(range(N_BATCHES), loader):
        print("doing batch", i)
        X = batch.view(FIT_PCA_BATCH_SIZE, -1)
        pca.partial_fit(X)
        bar.suffix = f"({(i+1)*FIT_PCA_BATCH_SIZE}/{N_BATCHES*FIT_PCA_BATCH_SIZE}) | Time: {time.time()-start:.1f}s | "
        bar.next()
    components = pca.mean_, pca.components_
    pickle.dump(components, open(GET_WEIGHTS_PATH(image_type),
                                 'wb'))
    bar.finish()

def make_latents(image_type, N0, N):
    print("making latents")
    pca = PCA()
    weights_path = GET_WEIGHTS_PATH(image_type)
    components_path = GET_COMPONENTS_PATH(image_type, N0, N)
    pca.mean_, pca.components_ = pickle.load(open(weights_path, 'rb'))
    loader = get_fake_loader(image_type,
                             FIT_PCA_BATCH_SIZE,
                             optional_flip=False,
                             N0=N0, N=N)
    bar =  Bar('Generating components', max=len(loader))
    start = time.time()
    for i, batch in enumerate(loader):
        latents = pca.transform(
            batch.view(FIT_PCA_BATCH_SIZE, -1)
        )
        if i == 0:
            all_latents = latents
        else:
            all_latents = np.concatenate((all_latents, latents), axis=0)
        # save progress after every batch
        pickle.dump(all_latents, open(components_path, 'wb'))
        bar.suffix = f"({(i+1)*FIT_PCA_BATCH_SIZE}/{len(loader)*FIT_PCA_BATCH_SIZE}) | Time: {time.time()-start:.1f}s | "
        bar.next()
    bar.finish()

def do_all(image_type):
    fit_pca(image_type)
    make_latents(image_type)

class BirdRegionPCA():
    def __init__(self, image_type, cuda,
                 N=-1):
        weights_path = GET_WEIGHTS_PATH(image_type)

        components_paths = glob
        component_pattern = GET_COMPONENTS_PATH(image_type,
                                                '*', '*')
        component_files = sorted(
            glob(component_pattern),
            key=lambda f: int(f.split('/')[-1]\
                               .split('.')[0]\
                               .split('_')[-1]\
                               .split('-')[0]),
        )
        self.comps = None
        for path in component_files:
            print(path)
            new_comps = pickle.load(open(path, 'rb'))
            self.comps = new_comps if self.comps is None else\
                    np.concatenate(
                        [self.comps,
                         new_comps],
                        axis=0)
            if N != -1 and len(self.comps) >= N:
                self.comps = self.comps[:N]
                break

        self.mean, self.weights = pickle.load(open(weights_path, 'rb'))
        self.mean, self.weights, self.comps = map(
            lambda t: torch.tensor(t).type(torch.FloatTensor),
            [self.mean, self.weights, self.comps])
        if cuda:
            self.mean = self.mean.cuda()
            self.weights = self.weights.cuda()
            self.comps = self.comps.cuda()

    def get_approx_regions(self, R, C,
                           n_chunks, flipped=False):
        """
        Returns attention patch at (R, C) with specified padding
        and flip. Generates it in n_chunks chunks to reduce
        memory requirements.
        """
        mean_slice = get_observed_patch(
            self.mean.view(1, 3, BIRD_IMG_DIM, BIRD_IMG_DIM),
            R, C, BIRD_ATT_DIM, horizontal_flip=flipped
        ).view(1, 3*BIRD_ATT_DIM**2)
        weights_slice = get_observed_patch(
            self.weights.view(-1, 3, BIRD_IMG_DIM, BIRD_IMG_DIM),
            R, C, BIRD_ATT_DIM, horizontal_flip=flipped
        ).view(-1, 3*BIRD_ATT_DIM**2)

        # return generator of reconstructed regions for chunks of images
        return ((mean_slice + torch.mm(chunk, weights_slice))\
                .view(-1, 3, BIRD_ATT_DIM, BIRD_ATT_DIM)
                for chunk in torch.chunk(self.comps, n_chunks))

    @allow_unbatched({1: [0]})
    def get_latents(self, images):
        images = images.flatten(start_dim=1)
        images = images - self.mean
        return torch.mm(images, self.weights.transpose(0, 1))

    def get_full_images(self, latents):
        unbatched = len(latents.shape) == 1
        if unbatched:
            latents = latents.unsqueeze(0)
        images = (self.mean + \
                  torch.mm(latents, self.weights)
                  ).view(-1, 3, BIRD_IMG_DIM, BIRD_IMG_DIM)
        if unbatched:
            images = images.squeeze(0)
        return images

    @allow_unbatched({1: [0]})
    def reconstruct(self, images):
        l = self.get_latents(images)
        return self.get_full_images(l)
