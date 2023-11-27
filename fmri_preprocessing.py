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
