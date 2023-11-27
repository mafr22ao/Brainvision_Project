
def PCA_and_save(activations_dir, save_dir, layers):
    """
    This function preprocesses Neural Network features using PCA and save the results
    in  a specified directory
.
    Parameters
    ----------
    activations_dir : str
        save path for extracted features.
    save_dir : str
        save path for extracted PCA features.
    layers: list
        a list of strings with layer names to perform pca
    """
    import os
    import numpy as np
    import tqdm
    import glob
    import time
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    seed = 42

    n_components = 100
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for layer in tqdm(layers):
        activations_file_list = glob.glob(activations_dir + '/*' + layer + '.npy')
        activations_file_list.sort()
        feature_dim = np.load(activations_file_list[0])
        x = np.zeros((len(activations_file_list), feature_dim.shape[0]))
        for i, activation_file in enumerate(activations_file_list):
            temp = np.load(activation_file)
            x[i, :] = temp
        x_train = x[:1000, :]
        x_test = x[1000:, :]

        start_time = time.time()
        x_test = StandardScaler().fit_transform(x_test)
        x_train = StandardScaler().fit_transform(x_train)
        ipca = PCA(n_components=n_components, random_state=seed)
        ipca.fit(x_train)

        x_train = ipca.transform(x_train)
        x_test = ipca.transform(x_test)
        train_save_path = os.path.join(save_dir, "train_" + layer)
        test_save_path = os.path.join(save_dir, "test_" + layer)
        np.save(train_save_path, x_train)
        np.save(test_save_path, x_test)

    print("This has not be implemented yet.")

PCA_and_save()
