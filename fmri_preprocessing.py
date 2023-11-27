def fmri_preprocessing():
    """
    conducts remaining preprocessing steps on the fmri's
    :return: saves the preprocessed files for later use
    """
    print("This has not be implemented yet.")

# voxel correlations (taken from CCN2021_Algonauts)
def vectorized_correlation(x,y):
    dim = 0

    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    covariance = covariance / (x.shape[dim])

    x_std = x.std(axis=dim, keepdims=True)+1e-8
    y_std = y.std(axis=dim, keepdims=True)+1e-8

    corr = covariance / (x_std * y_std)

    return corr.ravel()