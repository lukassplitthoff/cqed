from pathlib import Path
from qcodes import initialise_or_create_database_at, config, load_by_run_spec
from xarray import merge
import numpy as np
import scipy.linalg

import warnings


def create_local_dbase_in(folder_name='general', db_name='experiments.db', data_dir='D:/Data'):
    """    
    Initialise or create a QCoDeS database in D:/Data/folder_name/db_name 
    If the directory does not exist, it is created.
    Set the QCoDeS database location to point to the specified database.
    
    Inputs:
    folder_name (str): Name of the subfolder in D:/Data where db is stored. 
                       Can also be a path to a subfolder, e.g. general/sample1 leads to db in D:/Data/general/sample1 
    db_name (str):     Name of database, including .db
    
    data_dt (str):     Parent folder for all database files. 
    """
    
    dest_dir = Path(data_dir, folder_name)
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)        
    dest_path = Path(dest_dir, db_name)
    initialise_or_create_database_at(dest_path)
    config['core']['db_location'] = dest_path


def db_to_xarray(ind, **kwargs):

    """
    Take a dataset from a qcodes database identified by its ID and transform it into a xarray.Dataset
    Wraps around the function load_by_run_spec, which allows to get data from different databases, if you supply
    the corresponding connection. The snapshot of the dataset is available via the attrs attribute of the returned
    xarray.Dataset.

    @param ind: index of the dataset in the QCoDeS database you want to transform to a XArray
    @param kwargs: kwargs to be passed to the underlying qcodes function load_by_run_spec
    @return: xarray.Dataset with the independent parameters as coordinates,
     and the dependent parameters as Data variables
    """

    d = load_by_run_spec(captured_run_id=ind, **kwargs)
    ds = d.to_xarray_dataset()
    return ds



def max_variance_angle(data):
    """
    Find the angle of rotation for a complex dataset, which maximizes the variance of the data along the real axis
    @param data: (array) raw data assumed to be complex
    @return: (float) rotation angle that maximizes the variance in the real axis.
    """
    i = np.real(data)
    q = np.imag(data)
    cov = np.cov(i, q)
    a = scipy.linalg.eig(cov)
    eigvecs = a[1]

    if a[0][1] > a[0][0]:
        eigvec1 = eigvecs[:, 0]
    else:
        eigvec1 = eigvecs[:, 1]

    theta = np.arctan(eigvec1[0] / eigvec1[1])
    return theta


def rotate_data(data, theta):
    """
    Rotate the complex data by a given angle theta
    @param data: complex data to be rotated
    @param theta: angle of rotation (radians)
    @return: rotated complex array
    """

    return data * np.exp(1.j * theta)