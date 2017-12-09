import autograd.numpy as np
from autograd.scipy.special import gammaln, psi
import seaborn as sb
import matplotlib.pyplot as plt


def diag_gaussian_entropy(log_std, D):
    return 0.5 * D * (1.0 + np.log(2 * np.pi)) + np.sum(log_std)


def inv_gamma_entropy(a, b):
    return np.sum(a + np.log(b) + gammaln(a) - (1 + a) * psi(a))


def log_normal_entropy(log_std, mu, D):
    return np.sum(log_std + mu + 0.5) + (D / 2) * np.log(2 * np.pi)


def make_batches(n_data, batch_size):
    return [slice(i, min(i+batch_size, n_data)) for i in range(0, n_data, batch_size)]


