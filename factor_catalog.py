'''
To download pickled instances for FFHQ and LSUN-Bedrooms, visit: https://drive.google.com/open?id=1GYzEzOCaI8FUS6JHdt6g9UfNTmpO08Tt
'''

import torch
import ptutils
from spherical_kmeans import MiniBatchSphericalKMeans
from spherical_gmm import GMMWrapper
import numpy as np

def one_hot(a, n):
    b = np.zeros((a.size, n))
    b[np.arange(a.size), a] = 1
    return b

class FactorCatalog:

    def __init__(self, k, random_state=0, factorization=None, **kwargs):

        if factorization is None:
            factorization = MiniBatchSphericalKMeans

        self._factorization = factorization(k, random_state=random_state, **kwargs)

        self.annotations = {}

    def _preprocess(self, X, mask=None):
        X_flat = ptutils.partial_flat(X)
        if isinstance(mask, type(None)):
            return X_flat
        mask_flat = ptutils.partial_flat(mask)
        X_select = X_flat[mask_flat.view(-1)]
        return X_select

    def _postprocess(self, labels, X, raw, mask=None):
        if not isinstance(mask, type(None)):
            labels += 1
            mask_flat = ptutils.partial_flat(mask).numpy().reshape(-1)
            labels_ = np.zeros_like(mask_flat, dtype=labels.dtype)
            labels_[mask_flat] = labels
            heatmaps = torch.from_numpy(one_hot(labels_, self._factorization.cluster_centers_.shape[0]+1)).float()
        else:
            labels_= labels
            heatmaps = torch.from_numpy(one_hot(labels_, self._factorization.cluster_centers_.shape[0])).float()
        heatmaps = ptutils.partial_unflat(heatmaps, N=X.shape[0], H=X.shape[-1])
        if raw:
            heatmaps = ptutils.MultiResolutionStore(heatmaps, 'bilinear')
            return heatmaps
        else:
            heatmaps = ptutils.MultiResolutionStore(torch.cat([(heatmaps[:, v].sum(1, keepdim=True)) for v in
                        self.annotations.values()], 1), 'nearest')
            labels_ = list(self.annotations.keys())

            return heatmaps, labels_

    def fit_predict(self, X, raw=False, mask=None):
        self._factorization.fit(self._preprocess(X, mask))
        labels = self._factorization.labels_  
        #labels = self._factorization.fit_predict(self._preprocess(X))
        #self._factorization.cluster_centers_ = self._factorization.means_
        return self._postprocess(labels, X, raw, mask)

    def predict(self, X, raw=False, mask=None):
        labels = self._factorization.predict(self._preprocess(X, mask))
        return self._postprocess(labels, X, raw, mask)

    def __repr__(self):
        header = '{} catalog:'.format(type(self._factorization))
        return '{}\n\t{}'.format(header, self.annotations)