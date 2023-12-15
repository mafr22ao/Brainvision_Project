import os
import pickle
import numpy as np
import tensorflow as tf
import requests
import zipfile
import io

def download_fmri():
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

# def calculate_vectorized_correlation(x, y):
#     """
#     Calculate evaluation score (per voxel)
#     :param: x - prediction voxel activations; y - actual voxel activations
#     :return: evaluation score
#     Taken from challenge code from 2021 Algonauts challenge
#     """
#     dim = 0
#
#     centered_x = x - tf.reduce_mean(x, axis=dim, keepdims=True)
#     centered_y = y - tf.reduce_mean(y, axis=dim, keepdims=True)
#
#     covariance = tf.reduce_sum(centered_x * centered_y, axis=dim, keepdims=True)
#     covariance = covariance / tf.cast(tf.shape(x)[dim], tf.float32)
#
#     x_std = tf.math.reduce_std(x, axis=dim, keepdims=True) + 1e-8
#     y_std = tf.math.reduce_std(y, axis=dim, keepdims=True) + 1e-8
#
#     corr = covariance / (x_std * y_std)
#
#     return tf.reshape(corr, [-1])


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


def correlation_metric(y_true, y_pred):
    """
    makes "vectorized_correlation" usable as a keras metric for model validation & testing
    :param y_true:
    :param y_pred:
    :return:
    """

    correlation = tf.py_function(calculate_vectorized_correlation, [y_true, y_pred], tf.float32)
    return tf.reduce_mean(correlation)


def get_pca(layer, mode="val", import_type="direct"):
    """This function loads CNN features (preprocessed using PCA) into a
    numpy array according to a given layer.
    Parameters
    ----------
    :param layer : which layer of the neural network to load
    :param mode: "val" to get train & validation data, "test" to get test data
    Returns
    -------
    train_pca, val_pca, test_pca: PCA data after train-val-test split

    """

    # define directories for PCA and fmri data (repo_root necessary for runs in Ucloud)
    pca_dir = os.getcwd()

    if import_type == "direct":
        all_pcas = np.load(f"{layer}_pca.npy")
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
    train_pca = all_pcas[:800, :]
    val_pca = all_pcas[800:900, :]
    test_pca = all_pcas[900:, :]

    if mode == "val":
        print("train_pca shape: ", train_pca.shape)
        print("val_pca shape: ", val_pca.shape)
    elif mode == "test":
        print("test_pca shape: ", test_pca.shape)

    # standardize model inputs
    mean = np.mean(train_pca, axis=tuple(range(val_pca.ndim)))
    std_dev = np.std(train_pca, axis=tuple(range(train_pca.ndim)))

    if mode == "val":
        # Standardize train & validation
        train_pca = (train_pca - mean) / std_dev
        val_pca = (val_pca - mean) / std_dev

        return train_pca, val_pca
    elif mode == "test":
        # Standardize test
        test_pca = (test_pca - mean) / std_dev

        return test_pca
    else:
        print("Error: Unknown mode type")


def get_fmri(ROI, track, sub, mode="val"):
    """
    This function loads fMRI data into a numpy array for to a given ROI.
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
        print("ROI_test shape: ", ROI_test.shape)
        return ROI_test
    else:
        print("Error: Unknown mode type")
