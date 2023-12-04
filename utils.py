import os
import pickle


def find_repo_root(path='.'):
    """
    Find root path of repo.

    :param path: path of current directory where executing file is stored in.
    :return: path: root path of repo.
    """
    path = os.path.abspath(path)
    while not os.path.isdir(os.path.join(path, '.git')):
        parent = os.path.dirname(path)
        if parent == path:
            # We've reached the root of the file system without finding '.git'
            return None
        path = parent
    return path


def calculate_vectorized_correlation(x, y):
    """
    calculate evaluation score (per voxel)
    :param: x - prediction voxel activations; y - actual voxel activations
    :return: evaluation score
    """
    dim = 0

    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    covariance = covariance / (x.shape[dim])

    x_std = x.std(axis=dim, keepdims=True) + 1e-8
    y_std = y.std(axis=dim, keepdims=True) + 1e-8

    corr = covariance / (x_std * y_std)

    return corr.ravel()

