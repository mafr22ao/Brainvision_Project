import os
import pickle


def find_repo_root(path='.'):
    '''
    Find root path of repo.

    :param path: path of current directory where executing file is stored in.
    :return: path: root path of repo.
    '''
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

