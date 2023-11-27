def fmri_preprocessing():
    """
    conducts remaining preprocessing steps on the fmris
    :return: saves the preprocessed files for later use
    """
    import pickle
    def load_dict(filename_):
        with open(filename_, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            ret_di = u.load()
        return ret_di

    # path to ROI file
    ROI_file = "participants_data_v2021/full_track/sub04/WB.pkl"

    # loading .pkl file
    ROI_data = load_dict(ROI_file)
    print(ROI_data.keys())

    # print the data dimensions:
    print(ROI_data['train'].shape)
    print(ROI_data['voxel_mask'].shape)

    # data shape: (1000, 3, 19445) - three measurements for 1000 videos of subject4, who has 19445 voxels
    # mask shape: voxels are organized in a space of (78x93x71) voxels

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