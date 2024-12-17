from pathlib import Path
from typing import Optional, Union

import pandas as pd
import xarray as xr
from eoread.download_legacy import download_url
from core import env


def get_climate_data(dirname): 
    """
    Collect climate data from hygeos server 
    """
    download_url('https://docs.hygeos.com/s/5oWmND4xjmbCBtf/download/no2_climatology.hdf',dirname)
    download_url('https://docs.hygeos.com/s/4tzqH25SwK9iGMw/download/trop_f_no2_200m.hdf',dirname)


def get_absorption(gaz: str, dirname: Optional[Union[str, Path]]=None): 
    '''
    read absorption data for a gaz provided by user

    returns an array corresponding to the absorption rate of the gas as
    a function of wavelength [unit: nm]
    '''
    file_absorption_MTL = Path(__file__).parent/'absorption_MTL.csv'
    if dirname is None:
        dirname = env.getdir("DIR_STATIC")/"common"

    k_path = pd.read_csv(file_absorption_MTL, header=0)
    urlpath = k_path[k_path['gaz'] == gaz]['url_txt']
    skiprows = k_path[k_path['gaz'] == gaz]['skiprows'].values[0]
    if len(urlpath) == 0:
        raise FileNotFoundError(f'No corresponding file for gaz ({gaz})')
    
    txt_path = download_url(urlpath.values[0], dirname, verbose=False)
    abs_rate = pd.read_csv(txt_path, skiprows=skiprows, 
                    engine='python', dtype=float,
                    index_col=False, sep=' ',
                    names=["Wavelength", "Value"])
    k = xr.DataArray(abs_rate['Value'].to_numpy(), dims=["wav"])
    k = k.assign_coords(wav=abs_rate['Wavelength'].to_numpy())
    k['wav'].attrs.update({"units": "nm"})

    return k

