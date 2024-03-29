import numpy as np
import pickle
import os


def fmri_preprocessing():
    """
    conducts remaining preprocessing steps on the fmris
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


# get fmri data (taken from CCN2021_Algonauts)
def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
    return ret_di


def get_fmri(sub, ROI, avg=True):
    """
    Loads fMRI data for a specified ROI into a numpy array.

    Parameters:
    fmri_dir (str): Path to fMRI data.
    ROI (str): Name of the ROI.
    avg (bool, optional): If True, average data across repetitions. Defaults to True.

    Returns:
    np.array: fMRI response matrix (dimensions: #train_vids x #repetitions x #voxels).
    (Optional) np.array: Voxel mask for the 'WB' ROI.
    """
    fmri_dir = './participants_data_v2021'
    if ROI == "WB":
        track = "full_track"
    else:
        track = "mini_track"
    track_dir = os.path.join(fmri_dir, track)
    sub_fmri_dir = os.path.join(track_dir, sub)

    ROI_data = load_dict(os.path.join(sub_fmri_dir, f"{ROI}.pkl"))
    data = np.mean(ROI_data["train"], axis=1) if avg else ROI_data["train"]
    return (data, ROI_data['voxel_mask']) if ROI == "WB" else data
