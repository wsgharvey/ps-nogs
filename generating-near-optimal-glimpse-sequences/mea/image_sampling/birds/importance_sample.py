import itertools as it
import numpy as np
import torch

from ..importance_sample import ESS
from .pca import BirdRegionPCA
from .fake_loader import BatchlessFakeLoader
from mea.config import BIRD_ATT_DIM
from mea.utils import allow_unbatched, \
    get_observed_patch


class BirdImportanceSampler():
    def __init__(self, cuda,
                 n_chunks=10,
                 noise_var_p=1,
                 noise_var_q=1,
                 n_samples_p=100,
                 n_samples_q=200,
                 n_fake_images=-1):
        self.pca = BirdRegionPCA('images', cuda=cuda,
                                 N=n_fake_images)
        self.loader = BatchlessFakeLoader('images',
                                          optional_flip=True)
        self.device = 'cuda' if cuda else 'cpu'
        self.n_chunks = n_chunks
        self.noise_var_p = noise_var_p
        self.noise_var_q = noise_var_q
        self.n_samples_p = n_samples_p
        self.n_samples_q = n_samples_q

    def set_obs(self, true_image, locations):
        """
        locations should be ordered sequence of (r, c) tuples
        """
        self.locations = locations
        self.exact_obs = self.compute_observation(
            true_image, locations
        )
        approx_image = self.pca.reconstruct(true_image)
        self.approx_obs = self.compute_observation(
            approx_image, locations
        )

    @allow_unbatched({1: [0]})
    def compute_observation(self, images, locations):
        B = images.shape[0]
        return torch.cat(
            [get_observed_patch(
                images, r, c,
                BIRD_ATT_DIM, horizontal_flip=False
            ).view(B, -1)
             for r, c in locations],
            dim=1)

    def log_prob(self, x1, x2, noise_var):
        """
        computes vector of log probs for p(x1|x2). x2 may or may not be
        batched, x1 should not be (or have batch size 1).

        we assume shape for each is B x (n observed pixels)
        """
        assert len(x1.shape) in [1, 2]
        assert len(x2.shape) == 2
        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(0)
        diff = x1 - x2
        log_prob = -(diff**2).sum(dim=1)/noise_var
        return log_prob

    def get_logq(self):
        """
        returns categorical distribution. samples from this correspond
        to image that can be loaded by load_from_q
        """
        def get_approx_likelihood(flipped):
            region_seq = zip(*[
                self.pca.get_approx_regions(
                    r, c, n_chunks=self.n_chunks,
                    flipped=flipped)
                for r, c in self.locations])
            lps = []
            for regions in region_seq:
                all_regions = torch.cat(
                    [r.flatten(start_dim=1) for r in regions],
                    dim=1)
                lp = self.log_prob(self.approx_obs,
                                   all_regions,
                                   self.noise_var_q)
                lps.append(lp)
            return torch.cat(lps, dim=0)
        approx_loglikelihood = torch.cat(
            [get_approx_likelihood(False),
             get_approx_likelihood(True)],
            dim=0
        )
        q = approx_loglikelihood
        # store size so we can work out which were flipped later
        self.q_size = q.shape[0] // 2
        return q

    def get_logp(self, images):
        patches = self.compute_observation(images,
                                           self.locations)
        return self.log_prob(self.exact_obs,
                             patches.to(self.device),
                             self.noise_var_p)

    def get_indices(self, q_indices):
        indices = q_indices % self.q_size
        flipped = q_indices >= self.q_size
        return indices, flipped

    def load_from_q(self, q_indices):
        """
        loads images corresponding to indices sampled from distribution
        returned by get_q with flipped='both'
        """
        indices, flipped = self.get_indices(q_indices)
        return self.loader.load(indices, flipped).to(self.device)

    def resample(self, logw, n_samples):
        p = torch.softmax(logw, dim=0)
        expected_samples = p * n_samples
        samples = expected_samples // 1
        n_remain = int(n_samples - samples.sum())
        if n_remain > 0:
            remaining_expected_samples = expected_samples - samples
            remainder_probs = remaining_expected_samples \
                /remaining_expected_samples.sum()
            sampled_indices = np.random.choice(
                len(remainder_probs),
                size=n_remain,
                p=remainder_probs.to('cpu').numpy()
            )
            for index in sampled_indices:
                samples[index] += 1
        indices = torch.nonzero(samples).view(-1)
        indices = indices.to('cpu').numpy()
        counts = samples[indices]
        return indices, counts

    def unconditional_samples(self, return_indices):
        indices = torch.arange(self.n_samples_p)
        weights = (torch.ones(self.n_samples_p) \
                   / self.n_samples_p).to(self.device)
        flipped = torch.zeros(self.n_samples_p).bool()
        images = self.loader.load(indices, flipped)\
            .to(self.device)
        if return_indices:
            return images, weights, indices, flipped
        else:
            return images, weights

    def sample(self, true_image, locations,
               return_indices=False):
        if len(locations) == 0:
            return self.unconditional_samples(
                return_indices
            )
        self.set_obs(true_image, locations)

        # get q and then adjust noise variance so that it has good ESS
        self.noise_var_q = 1.
        logq = self.get_logq()

        # do a binary search for a noise variance that gives good ESS
        min_scaler = -6.
        max_scaler = 6.
        ess_logq, _ = logq.topk(min(20*self.n_samples_q, len(logq)))  # highest-weight samples used to quickly approximate effective sample size
        ess_logq = ess_logq.detach().cpu()
        while True:
            scaler = (min_scaler+max_scaler)/2
            ess = ESS(
                torch.softmax(
                    ess_logq*10**scaler,
                    dim=-1)
            )
            if ess > self.n_samples_q:
                min_scaler = scaler
            elif ess < 0.8*self.n_samples_q:
                max_scaler = scaler
            else:
                noise_var = self.noise_var_q/scaler
                break
        logq = logq*10**scaler

        q_indices, counts = self.resample(logq, self.n_samples_q)
        images = self.load_from_q(q_indices)
        logw = counts.log()
        # samples from q are now represented by images, logw
        logp = self.get_logp(images)
        logw = logw + logp - logq[q_indices]
        # samples reweighted to approximate p, represented by images, logw
        p_indices, counts = self.resample(logw, self.n_samples_p)
        images = images[p_indices]
        normalised_weights = counts / self.n_samples_p
        if return_indices:
            indices, flipped = self.get_indices(q_indices)
            return images, normalised_weights, \
                indices, flipped
        else:
            return images, normalised_weights
