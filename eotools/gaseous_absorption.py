from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from eoread.download import download_url


def get_climate_data(dirname): 
    """
    Collect climate data from hygeos server 
    """
    download_url('https://docs.hygeos.com/s/5oWmND4xjmbCBtf/download/no2_climatology.hdf',dirname)
    download_url('https://docs.hygeos.com/s/4tzqH25SwK9iGMw/download/trop_f_no2_200m.hdf',dirname)


def get_absorption(gaz:str, dirname): 
    '''
    read absorption data for a gaz provided by user

    returns an array corresponding to the absorption rate of the gas as
    a function of wavelength [unit: nm]
    '''
    file_absorption_MTL = Path(__file__).parent/'absorption_MTL.csv'
    k_path = pd.read_csv(file_absorption_MTL, header=0)
    urlpath = k_path[k_path['gaz'] == gaz]['url_txt']
    skiprows = k_path[k_path['gaz'] == gaz]['skiprows'].values[0]
    if len(urlpath) == 0:
        raise FileNotFoundError(f'No corresponding file for gaz ({gaz})')
    
    txt_path = download_url(urlpath.values[0], dirname)
    abs_rate = pd.read_csv(txt_path, skiprows=skiprows, 
                    engine='python', dtype=float,
                    index_col=False, sep=' ',
                    names=["Wavelength", "Value"])
    return xr.DataArray(abs_rate['Value'].to_numpy(),
                        coords={'wav':abs_rate['Wavelength'].to_numpy()})


def combine_with_srf(srf: xr.Dataset, lut:xr.DataArray):
    """
    Compute the SRF-weighted integration of the variable lut
    
    Arguments:
        - srf : an xr.Dataset of the srf returned by get_SRF
        - lut : an xr.DataArray of a specific variable
    
    Return a dictionary containing the integration value for each wavelength
    """
    output_dic = {}
    for band in srf:
        srf_band = srf[band].rename({f'wav_{band}':'wav'})
        lut_interp = lut.interp(wav=srf_band.coords['wav'].values)
        integrate = np.trapz(lut_interp*srf_band, x=lut_interp.coords['wav'])
        normalize = np.trapz(srf_band, x=lut_interp.coords['wav'])
        output_dic[band] = integrate/normalize
    return output_dic