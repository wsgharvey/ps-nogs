import itertools as it
import time
import numpy as np
import torch
from .fake_loader import BatchlessFakeLoader
from ..config import IMG_DIM, ATT_DIM
from .pca import RegionPCA
from ..utils import take_patch, transformed_coords
import logging
log = logging.Logger(__name__)
log.setLevel(logging.ERROR)


def ESS(w):
    """
    computes effective sample size given possibly unnormalised weights (but not log weights)
    """
    return sum(w)**2/np.square(w).sum()


def normalise_img(x, normalise_std):
    """ per-channel normalisation
    """
    B, C = x.shape[:2]
    img_shape = x.shape[2:]
    x = x.view(B, C, -1)
    x = x - x.mean(dim=2, keepdim=True)
    if normalise_std:
        x = x / (x.std(dim=2, keepdim=True) + 0.001)
    x = x.view((B, C) + img_shape)
    return x


class ImportanceSampler():
    def __init__(self, cuda, test=False):
        self.pca = RegionPCA(cuda=cuda, test=test)
        self.loader = BatchlessFakeLoader(optional_flip=True)
        self.cuda = cuda

    def set_obs(self, true_image, locations):
        """
        self.obs is list of size (1, 3, ATT_DIM, ATT_DIM) for each location
        """
        img = true_image.unsqueeze(0)
        self.exact_obs = self.compute_observation(img, locations)
        approx_img = self.pca.reconstruct(img)
        self.approx_obs = self.compute_observation(approx_img, locations)

    def compute_observation(self, batch_img, locations, padding=0):
        assert len(batch_img.shape) == 4
        obs = []
        EXTRACT_DIM = ATT_DIM + padding*2
        for r, c in locations:
            b = batch_img[:, :, r:r+EXTRACT_DIM, c:c+EXTRACT_DIM]
            b = b.contiguous()
            obs.append(b)
        return obs

    def translate_images(self, images, coords):
        """
        in-place translation of images by coords
        """
        for idx, (image, (R, C)) in enumerate(zip(images, coords)):

            image = take_patch(image, IMG_DIM, R, C, IMG_DIM)

    def log_prob(self, x1, x2, noise_var, normalise):
        """ computes vector of log probs for p( x1 | x2 ) assuming Gaussian noise
            either or both x1 and x2 can be batched
        """

        if normalise:
            x1 = normalise_img(x1, normalise_std=False)
            x2 = normalise_img(x2, normalise_std=False)

        diff = x1 - x2
        diff = diff.flatten(start_dim=1)
        log_prob = -(diff**2).sum(dim=1)/noise_var
        return log_prob

    def convolve_log_prob(self, x1, x2, noise_var,
                          max_shift, stride, marginalise_shift,
                          normalise):
        """ x2 must be the bigger one
        """

        indices = np.array(list(it.product(range(0, (x2.shape[-2]+1-ATT_DIM), stride),
                                           range(0, (x2.shape[-1]+1-ATT_DIM), stride))))

        def get_log_prob(r, c): return self.log_prob(x1, x2[:, :, r:r+ATT_DIM, c:c+ATT_DIM],
                                                     noise_var, normalise=normalise)
        B = x2.shape[0]
        if marginalise_shift:
            log_probs = -torch.ones(B)/0  # -inf
            log_probs = log_probs.cuda() if self.cuda else log_probs
        else:
            t = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
            log_probs = t(B, len(indices))

        for i, (r, c) in enumerate(indices):
            if marginalise_shift:
                torch.logsumexp(torch.stack([log_probs, get_log_prob(r, c)], dim=0),
                                dim=0,
                                out=log_probs)
            else:
                log_probs[:, i] = get_log_prob(r, c)

        if marginalise_shift:
            return log_probs
        else:
            def get_img_coord(coord_index):
                return indices[coord_index.numpy()]
            return log_probs, get_img_coord

    def get_q(self, locations, noise_var, normalise,
              max_shift, stride, marginalise_shift,
              n_chunks):

        im_regions_f = []
        im_regions_nf = []
        for r, c in locations:
            ims_r_c_f = self.pca.get_approx_regions(r, c, n_chunks, flipped=True, padding=max_shift)
            ims_r_c_nf = self.pca.get_approx_regions(r, c, n_chunks, flipped=False, padding=max_shift)
            im_regions_f.append(ims_r_c_f)
            im_regions_nf.append(ims_r_c_nf)

        full_q = []
        for im_regions in [im_regions_nf, im_regions_f]:
            qs = []
            for im_regions_chunk in zip(*im_regions):
                q = 0
                for im_region, obs_region in zip(im_regions_chunk, self.approx_obs):
                    q_region = self.convolve_log_prob(obs_region, im_region, noise_var,
                                                      max_shift=max_shift, stride=stride,
                                                      marginalise_shift=marginalise_shift,
                                                      normalise=normalise)
                    q = q + q_region
                qs.append(q)
            full_q.append(torch.cat(qs, dim=0))
        q = torch.stack(full_q, dim=1)  # now -1 x |FLIP| x |LOCATIONS|

        def get_flip(flip_index): return flip_index == 1

        return q, get_flip

    def load_images(self, indices, flipped):
        self.loader.batch_sampler.sampler = zip(indices, flipped)
        batch = next(self.loader.__iter__())
        return batch

    def get_p(self, images, locations, noise_var, normalise,
              max_shift, stride, marginalise_shift,
              n_chunks):
        assert marginalise_shift, "non-marginalised not implemented for p"
        ps = []
        for images_chunk in torch.chunk(images, n_chunks):
            patches = self.compute_observation(images_chunk, locations, max_shift)
            p = None
            for patch, patch_obs in zip(patches, self.exact_obs):
                patch_log_prob = self.convolve_log_prob(patch_obs, patch, noise_var,
                                                        max_shift, stride, marginalise_shift,
                                                        normalise)
                p = patch_log_prob if p is None else \
                    torch.logsumexp(torch.stack([p, patch_log_prob], dim=0), dim=0)
            ps.append(p)
        return torch.cat(ps)

    def resample(self, log_w, n_samples):
        p = torch.softmax(log_w, dim=0)
        expected_samples = p*n_samples
        sampled = (expected_samples//1)
        n_remain = int(n_samples - sampled.sum())
        if n_remain > 0:
            remaining_expected_samples = expected_samples - sampled
            remainder_dist_probs = remaining_expected_samples/remaining_expected_samples.sum()  # n_remain down to numerical error
            if self.cuda:
                remainder_dist_probs = remainder_dist_probs.cpu()
            # use numpy dist as it is much faster for large prob vectors
            sampled_indices = np.random.choice(len(remainder_dist_probs),
                                               size=n_remain,
                                               p=remainder_dist_probs.numpy())
            for index in sampled_indices:
                sampled[index] += 1
        indices = torch.nonzero(sampled).view(-1)
        indices = indices.cpu().numpy() if self.cuda else indices.numpy()
        counts = sampled[indices]
        return indices, counts

    def posterior_samples(self, true_image, locations, n_p_samples,
                          n_q_samples, noise_var, n_chunks_q, max_shift_q,
                          stride_q, n_chunks_p, max_shift_p, stride_p,
                          normalise=False, return_indices=False):
        """
        N1: number of samples to draw from proposal distribution
        N2: number of samples to represent final posterior
        """

        start = time.time()

        if len(locations) == 0:
            # return N2 images from prior if we don't condition
            indices = np.random.choice(len(self.pca.comps), size=200)
            flips = np.random.random(size=200) < 0.5
            images = self.loader.load(indices, flipped=flips)
            weights = torch.ones(n_p_samples) / n_p_samples
            if self.cuda:
                images = images.cuda()
                weights = weights.cuda()
            if return_indices:
                return images, weights, indices, flips
            else:
                return images, weights

        self.set_obs(true_image, locations)

        # compute proposal distribution
        full_q, get_flip = self.get_q(locations, noise_var,
                                      normalise, max_shift_q,
                                      stride_q, True,
                                      n_chunks_q)
        flat_q = full_q.view(-1)

        # sample from proposal
        q_sampled_indices, counts = self.resample(flat_q, n_q_samples)

        # compute weights of samples
        q = flat_q[q_sampled_indices]

        # reshape indices into img_index x locations x flip
        q_sampled_img_indices, flips = transformed_coords(flat_q.shape, full_q.shape,
                                                          q_sampled_indices.reshape(1, -1))
        q_sampled_flipped = get_flip(flips)

        images = self.loader.load(q_sampled_img_indices.numpy(), q_sampled_flipped)
        if self.cuda:
            images = images.cuda()

        # reweight and resample --------------------------------------------
        p = self.get_p(images, locations, noise_var, normalise,
                       max_shift_p, stride_p, True,
                       n_chunks_p)
        log_w = p - q + counts.log()
        # resample to get rid of low weight samples
        p_sampled_indices, counts = self.resample(log_w, n_p_samples)
        images = images[p_sampled_indices]
        weights = counts/n_p_samples

        if return_indices:
            sample_indices = q_sampled_img_indices[p_sampled_indices]
            sample_flips = q_sampled_flipped[p_sampled_indices]

        if return_indices:
            sample_indices = q_sampled_img_indices[p_sampled_indices]
            sample_flips = q_sampled_flipped[p_sampled_indices]
            return images, weights, sample_indices, sample_flips
        else:
            return images, weights
