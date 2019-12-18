from pathlib import Path
from qcodes import initialise_or_create_database_at, config


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