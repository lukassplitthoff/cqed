from pathlib import Path
from qcodes import initialise_or_create_database_at, config, load_by_run_spec
from xarray import merge


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
    _df = []
    for obj in d.dependent_parameters:
        _df += [d.to_pandas_dataframe_dict()[obj.name].to_xarray()]

    ds = merge([*_df])
    ds.attrs['snapshot'] = d.snapshot
    ds.attrs['exp_name'] = d.exp_name
    ds.attrs['captured_run_id'] = d.captured_run_id
    ds.attrs['sample_name'] = d.sample_name
    ds.attrs['guid'] = d.guid
    ds.attrs['run_timestamp_raw'] = d.run_timestamp_raw
    ds.attrs['completed_timestamp_raw'] = d.completed_timestamp_raw

    return ds