import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import load_model
from utils import calculate_vectorized_correlation, get_pca, get_fmri


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


def test_model(model_name, layer, ROI, sub, X_test, y_test, df, mode="test"):
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

    :return: overview over correlation score values
    """
    print("testing model: ", model_name)

    # navigate to correct stored model
    model_dir = os.path.join(os.getcwd(), "models", layer, ROI, sub, model_name)
    model = load_model(model_dir)

    # extract hyperparameter settings from model_name
    split_string = model_name.split('_')
    num_hidden_layers = int(split_string[2])
    learning_rate = float(split_string[4])
    dropout = float(split_string[6])
    l2_reg = float(split_string[8].replace(".keras", ""))

    # calculate predicted voxel activations
    prediction = model.predict(X_test)

    # calculate evaluation metric
    test_corr = calculate_vectorized_correlation(y_test, prediction)

    # add evaluation metric results to the results dataframe
    new_values = {'stage': layer,
                  'ROI': ROI,
                  'sub': sub,
                  'correlation_score': test_corr}
    new_values = pd.DataFrame(new_values)
    new_values['voxel'] = new_values.index + 1
    new_values['num_hidden_layers'] = num_hidden_layers
    new_values['learning_rate'] = learning_rate
    new_values['dropout'] = dropout
    new_values['l2_reg'] = l2_reg

    df = pd.concat([df, new_values], ignore_index=True)

    # save the predicted fmri's
    if mode == "test":
        predictions_dir = os.path.join(os.getcwd(), "predictions", layer, ROI, sub)
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        np.save(f'prediction_hidden_{num_hidden_layers}_lr_{learning_rate}_dropout_{dropout}_l2_{l2_reg}".npy',
                prediction)

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
    column_names = ['voxel', 'stage', 'ROI', 'sub',
                    'num_hidden_layers', 'learning_rate', 'dropout', 'l2_reg',
                    'correlation_score']
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
            model_path = os.path.join("models", layer, ROI, sub)
            try:
                if data_mode == "test":
                    y_test = get_fmri(ROI, track, sub, mode=data_mode)

                    for model_name in os.listdir(model_path):
                        test_results = test_model(model_name, layer, ROI, sub, X_test, y_test, test_results, data_mode)
                elif data_mode == "val":
                    y_train, y_val = get_fmri(ROI, track, sub, mode=data_mode)
                    print(len(list(set(os.listdir(model_path)))))
                    for model_name in os.listdir(model_path):
                        test_results = test_model(model_name, layer, ROI, sub, X_val, y_val, test_results, data_mode)
            except OSError:
                print(f"Execution for {sub} ended at {ROI}")
                break
            print(f"finished testing sub: {sub}, ROI: {ROI}")

    # calculate aggregated scores

    # aggregate per subject
    test_results_aggregated = test_results.groupby(["ROI", "stage", "sub",
                                                    'num_hidden_layers', 'learning_rate', 'dropout', 'l2_reg']
                                                   )["correlation_score"].agg(
                                                    np.mean).reset_index()
    # aggregate over subjects
    test_results_aggregated = test_results_aggregated.groupby(["ROI", "stage",
                                                               'num_hidden_layers', 'learning_rate',
                                                               'dropout', 'l2_reg',
                                                              ])["correlation_score"].agg(
                                                                np.mean).reset_index()

    # save the dataframes
    save_test_results(test_results, test_results_aggregated, layer, data_mode)
