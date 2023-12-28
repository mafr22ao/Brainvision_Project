import os
import pickle
import numpy as np
import tensorflow as tf
import requests
import zipfile
import io
import tensorflow as tf
from keras.models import load_model
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
    :param import_type: either "direct" or indirect". Specifies if one single PCA file got loaded directly
                        into the CWD, or there are separate files per video in a folder structure
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
    base_train = all_pcas[:800, :]
    base_val = all_pcas[800:900, :]
    base_test = all_pcas[900:, :]

    if mode == "val":
        print("base_train shape: ", base_train.shape)
        print("base_val shape: ", base_val.shape)
    elif mode == "test":
        print("base_test shape: ", base_test.shape)

    # detect motion feature type
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

    # standardize motion inputs
    motion_mean = np.mean(motion_train, axis=tuple(range(motion_train.ndim)))
    motion_std_dev = np.std(motion_train, axis=tuple(range(motion_train.ndim)))

    if mode == "val":
        # Standardize train & validation
        base_train = (base_train - mean) / std_dev
        base_val = (base_val - mean) / std_dev
        motion_train = (motion_train - motion_mean) / motion_std_dev
        motion_val = (motion_val - motion_mean) / motion_std_dev

        pca_train = np.concatenate((base_train, motion_train), axis=1)
        pca_val = np.concatenate((base_val, motion_val), axis=1)

        return pca_train, pca_val
    elif mode == "test":
        # Standardize test
        base_test = (base_test - mean) / std_dev
        motion_test = (motion_test - motion_mean) / motion_std_dev

        pca_test = np.concatenate((base_test, motion_test), axis=1)

        return pca_test
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

    fmri_dir = os.path.join(os.getcwd(), "../participants_data_v2021", track, sub)

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


def save_test_results(df1, df2, layer, data_mode="test"):
    """
    saves test results for testing / validation in csv files
    :param df1: previously created df with one line per voxel, subject, ROI & layer
    :param df2: previously created df aggregated over voxels and subject (yields score per ROI & layer)
    :param layer: used stage of the feature extraction model
    :param data_mode: set to "val" if validation scores have been calculated
    """
    if data_mode == "val":
        mode_str = "_val"
    else:
        mode_str = ""
    test_scores_dir = os.path.join(os.getcwd(), "test_scores")
    if not os.path.exists(test_scores_dir):
      os.makedirs(test_scores_dir)
    df1.to_csv(os.path.join(test_scores_dir, f"test_results_{layer}{mode_str}.csv"), index=False)
    df2.to_csv(os.path.join(test_scores_dir, f"test_results_aggregated_{layer}{mode_str}.csv"), index=False)


def test_model(layer, ROI, sub, X_test, y_test, df, mode="test"):
    """
    reads in model for a certain stage, ROI & subject
    tests the models, saves the predicted brain activations, and appends the test results to the results df
    :param layer: used stage of the feature extraction model
    :param ROI: region of interest
    :param sub: current subject
    :param X_test: test data. Use validation for mode "val" and test for mode "test"
    :param y_test: test labels. Use validation for mode "val" and test for mode "test"
    :param df: df containing previous testing results
    :param mode:  set to "val" if validation scores have been calculated
    """
    """

    :return: overview over correlation score values
    """
    # navigate to correct stored model
    model_dir = os.path.join(os.getcwd(), "models", layer, ROI, sub)
    model = load_model(model_dir + "/model.keras")

    # calculate predicted voxel activations
    prediction = model.predict(X_test)

    # calculate evaluation metric
    test_corr = calculate_vectorized_correlation(y_test, prediction)

    # add evaluation metric results to the results dataframe
    new_values = {'stage': layer, 'ROI': ROI, 'sub': sub, 'correlation_score': test_corr}
    new_values = pd.DataFrame(new_values)
    new_values['voxel'] = new_values.index + 1
    df = pd.concat([df, new_values], ignore_index=True)

    # save the predicted fmri's
    if mode == "test":
        predictions_dir = os.path.join(os.getcwd(), "predictions", layer, ROI, sub)
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        np.save('../prediction.npy', prediction)

    return df


def run_evaluation_pipeline(data_mode="test"):
    """
    evaluates previously trained and saved models. Saves detailed & aggregated summaries of correlation score as csv, and the predicted brain activations
    :param data_mode: set to "test" for regular testing, and to "val" to get validation scores based on validation set
    """
    # load one only one main PCA file into the Ucloud session. This will determine the layer
    layer_list = ["stage_1", "stage_2", "stage_3", "stage_4", "stage_5", "final"]
    for i in layer_list:
        if os.path.exists(f"{i}_pca.npy"):
            layer = i
            break

    subs = ["sub01", "sub02", "sub03", "sub04", "sub05", "sub06", "sub07", "sub08", "sub09", "sub10"]
    ROIs = ["WB", "V1", "V2", "V3", "V4", "LOC", "EBA", "FFA", "STS", "PPA"]

    # test results dataframe
    column_names = ['voxel', 'stage', 'ROI', 'sub', 'correlation_score']
    test_results = pd.DataFrame(columns=column_names)

    # get test data
    if data_mode == "test":
        X_test = get_pca(layer, mode=data_mode)
    elif data_mode == "val":
        X_train, X_val = get_pca(layer, mode=data_mode)

    for sub in subs:
        for ROI in ROIs:
            # read in test data
            if ROI == "WB":
                track = "full_track"
            else:
                track = "mini_track"
            try:
                if data_mode == "test":
                    y_test = get_fmri(ROI, track, sub, mode=data_mode)
                    test_results = test_model(layer, ROI, sub, X_test, y_test, test_results, data_mode)
                elif data_mode == "val":
                    y_train, y_val = get_fmri(ROI, track, sub, mode=data_mode)
                    test_results = test_model(layer, ROI, sub, X_val, y_val, test_results, data_mode)
            except OSError:
                print(f"Execution for {sub} ended at {ROI}")
                break
            print(f"finished testing sub: {sub}, ROI: {ROI}")

    # calculate aggregated scores

    # aggregate per subject
    test_results_aggregated = test_results.groupby(["ROI", "stage", "sub"])["correlation_score"].agg(
        np.mean).reset_index()
    # aggregate over subjects
    test_results_aggregated = test_results_aggregated.groupby(["ROI", "stage"])["correlation_score"].agg(
        np.mean).reset_index()

    # save the dataframes
    save_test_results(test_results, test_results_aggregated, layer, data_mode)
