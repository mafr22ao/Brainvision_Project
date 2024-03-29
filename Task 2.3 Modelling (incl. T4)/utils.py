import os
import pickle
import numpy as np
import tensorflow as tf
import requests
import zipfile
import io


def download_fmri():
    """
    Downloads the fMRI data from dropbox link
    adapted from the 2021 Algonauts challenge code
    """
    dropbox_link = 'https://www.dropbox.com/s/agxyxntrbwko7t1/participants_data.zip?dl=1'

    if dropbox_link:
      fname = 'participants_data_v2021'
      if not os.path.exists(fname):
        print('Data downloading...')
        r = requests.get(dropbox_link)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        print('Data download is completed.')
      else:
        print('Data are already downloaded.')
    else:
      print('You need to submit the form and get the dropbox link')


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


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
    return ret_di


def calculate_vectorized_correlation(x, y):
    """
    calculate evaluation score (per voxel)
    :param: x - prediction voxel activations; y - actual voxel activations
    :return: evaluation score
    taken from challenge code from 2021 Algonauts challenge
    """
    from numpy import mean, std
    dim = 0

    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    covariance = covariance / (x.shape[dim])

    x_std = x.std(axis=dim, keepdims=True) + 1e-8
    y_std = y.std(axis=dim, keepdims=True) + 1e-8

    corr = covariance / (x_std * y_std)

    return corr.ravel()


def get_pca(layer, mode="val", import_type="direct", motion=True):
    """This function loads CNN features (preprocessed using PCA) into a
    numpy array according to a given layer.
    Parameters
    ----------
    :param layer : which layer of the neural network to load
    :param mode: "val" to get train & validation data, "test" to get test data
    :param import_type: either "direct" or indirect". Specifies if one single PCA file got loaded directly
                        into the CWD, or there are separate files per video in a folder structure
    :param motion: True/False, determines whether motion will be used as additional data input
    Returns
    -------
    train_pca & val_pca or test_pca: PCA data after train-val-test split. Includes motion data if motion=True

    """

    # define directories for PCA and fmri data (repo_root necessary for runs in Ucloud)
    pca_dir = os.getcwd()

    if import_type == "direct":
        try:
            with open(f"{layer}_pca.pkl", 'rb') as file:
                all_pcas = pickle.load(file)
                if isinstance(all_pcas, dict):
                    all_pcas = np.array(list(all_pcas.values()))
        except:
            all_pcas = np.load(f"{layer}_pca.npy")

        # reshape
        all_pcas = all_pcas.reshape(1000, -1)
    else:
        # numpy arrays of the PCA results
        all_pcas = []
        stage_path = os.path.join(pca_dir, layer)
        # Loop through each file in the folder
        for filename in os.listdir(stage_path):
            file_path = os.path.join(stage_path, filename)
            with open(file_path, 'rb') as file:
                loaded_data = pickle.load(file)
                # Convert loaded data to NumPy array if needed
                if isinstance(loaded_data, np.ndarray):
                    # Add a new axis before appending to the list
                    loaded_data = loaded_data[np.newaxis, ...]
                    all_pcas.append(loaded_data)
                else:
                    # Convert to array if data is not already in array format
                    loaded_data = np.array(loaded_data)
                    # Add a new axis before appending to the list
                    loaded_data = loaded_data[np.newaxis, ...]
                    all_pcas.append(loaded_data)

        # Concatenate the data along the new axis (axis=0 for a new dimension)
        all_pcas = np.concatenate(all_pcas, axis=0)

        # flatten PCA data over all dimensions but the first
        all_pcas = all_pcas.reshape(1000, -1)

    # Creating data splits
    base_train = all_pcas[:800, :]
    base_val = all_pcas[800:900, :]
    base_test = all_pcas[900:, :]

    if mode == "val":
        print("base_train shape: ", base_train.shape)
        print("base_val shape: ", base_val.shape)
    elif mode == "test":
        print("base_test shape: ", base_test.shape)

    if motion:
        # detect motion feature type (different types tried out during the project)
        motion_types = ["layer4", "avgpool", "stacked"]
        for i in motion_types:
            if os.path.exists(f"train_{i}.npy"):
                motion_feature_type = i
                break

        # load motion features
        if mode == "val":
            motion_train = np.load(f"train_{motion_feature_type}.npy")
            motion_val = np.load(f"val_{motion_feature_type}.npy")
            print("motion_train shape: ", motion_train.shape)
            print("motion_val shape: ", motion_val.shape)
        elif mode == "test":
            motion_train = np.load(f"train_{motion_feature_type}.npy")
            motion_test = np.load(f"test_{motion_feature_type}.npy")

    # standardize model inputs
    mean = np.mean(base_train, axis=tuple(range(base_train.ndim)))
    std_dev = np.std(base_train, axis=tuple(range(base_train.ndim)))

    if motion:
        # standardize motion inputs
        motion_mean = np.mean(motion_train, axis=tuple(range(motion_train.ndim)))
        motion_std_dev = np.std(motion_train, axis=tuple(range(motion_train.ndim)))

    if mode == "val":
        # Standardize train & validation
        base_train = (base_train - mean) / std_dev
        base_val = (base_val - mean) / std_dev
        if motion:
            motion_train = (motion_train - motion_mean) / motion_std_dev
            motion_val = (motion_val - motion_mean) / motion_std_dev
            pca_train = np.concatenate((base_train, motion_train), axis=1)
            pca_val = np.concatenate((base_val, motion_val), axis=1)
        else:
            pca_train = base_train
            pca_val = base_val

        return pca_train, pca_val
    elif mode == "test":
        # Standardize test
        base_test = (base_test - mean) / std_dev
        if motion:
            motion_test = (motion_test - motion_mean) / motion_std_dev
            pca_test = np.concatenate((base_test, motion_test), axis=1)
        else:
            pca_test = base_test

        return pca_test
    else:
        print("Error: Unknown mode type")


def get_fmri(ROI, track, sub, mode="val"):
    """
    This function retrieves the fMRI data for a given ROI & subject
    Inspired by the 2021 Algonauts challenge code
    Parameters
    ----------
    :param ROI: ROI of interest
    :param track : which track the ROI belongs to (mini or full)
    :param sub: subject of interest
    :param mode: "val" to get train & validation data, "test" to get test data
    Returns
    -------
    train & validation / test fmri data as arrays
    """

    fmri_dir = os.path.join(os.getcwd(), "participants_data_v2021", track, sub)

    # Loading ROI data
    ROI_file = os.path.join(fmri_dir, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    # averaging ROI data across repetitions
    ROI_data = np.mean(ROI_data["train"], axis=1)

    # flatten ROI data over 2nd and 3rd dimension
    ROI_data = ROI_data.reshape(1000, -1)

    ROI_train = ROI_data[:800]
    ROI_val = ROI_data[800:900]
    ROI_test = ROI_data[900:]

    if mode == "val":
        print("ROI_train shape: ", ROI_train.shape)
        print("ROI_val shape: ", ROI_val.shape)
        return ROI_train, ROI_val
    elif mode == "test":
        # print("ROI_test shape: ", ROI_test.shape)
        return ROI_test
    else:
        print("Error: Unknown mode type")
