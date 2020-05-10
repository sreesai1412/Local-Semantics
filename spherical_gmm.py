import warnings
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.cluster._k_means_fast import _mini_batch_update_csr
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster.k_means_ import (
    _init_centroids,
    _labels_inertia,
    _tolerance,
    _mini_batch_step,
    _mini_batch_convergence
)
from sklearn.mixture import GaussianMixture

class GMMWrapper(GaussianMixture):
    def __init__(self, n_components, random_state=0, **kwargs):
        super().__init__(n_components, random_state=random_state, **kwargs)
        self.labels_ = None
        self.cluster_centers_ = None