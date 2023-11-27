def data_load():
    """
    downloads the data from Dropbox. Code from the
    :return: saves the files in the "participants_data_v2021" folder
    """
    # @title Run the cell to download the data

    import requests, zipfile, io, os

    # Use the dropbox link to download the data
    dropbox_link = 'https://www.dropbox.com/s/agxyxntrbwko7t1/participants_data.zip?dl=1'
    if dropbox_link:
        fname1 = 'participants_data_v2021'
        fname2 = 'AlgonautsVideos268_All_30fpsmax'
        if not os.path.exists(fname1) or not os.path.exists(fname2):
            print('Data downloading...')
            r = requests.get(dropbox_link)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall()
            print('Data download is completed.')
        else:
            print('Data Load: Data are already downloaded.')

        url = 'https://github.com/Neural-Dynamics-of-Visual-Cognition-FUB/Algonauts2021_devkit/raw/main/example.nii'
        fname = 'example.nii'
        if not os.path.exists(fname):
            r = requests.get(url, allow_redirects=True)
            with open(fname, 'wb') as fh:
                fh.write(r.content)
        else:
            print(f"Data Load: {fname} file is already downloaded.")

    else:
        print('Data Load: You need to submit the form and get the dropbox link')