import numpy as np
import pandas as pd
import xarray as xr

import os
import tarfile
from glob import glob

from .download import download_url


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
    k_path = pd.read_csv('eotools/ancillary/absorption_MTL.csv', header=0)
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


def get_SRF(sensor:str, platform:str, dirpath:str = 'auxdata/srf/'):
    """
    Download and store SRF from EUMETSAT Website
        -> https://nwp-saf.eumetsat.int/site/software/rttov/download/coefficients/spectral-response-functions/
    
    Arguments:
        - sensor [str]   : name of the sensor in level1 reader
        - platform [str] : name of the platform in level1 reader
        - dirpath [str]  : directory path where to save SRF files
    
    Return a xr.DataArray with the different SRF 
    """
    tar_gz_path = pd.read_csv('eotools/ancillary/srf_eumetsat.csv', header=0, index_col=0)
    urlpath = tar_gz_path[((tar_gz_path['id_sensor'] == sensor) & (tar_gz_path['id_platform'] == platform))]['url_tar']
    if len(urlpath) == 0:
        raise FileNotFoundError(f'No corresponding file for sensor ({sensor}). \
            If {sensor} is a new type of data read by eoread module, please update csv file')
    urlpath = urlpath.values[0]

    ds = xr.Dataset()
    basename = os.path.basename(urlpath).split('.')[0]
    if not os.path.exists(dirpath+basename):
        tar_gz_path = download_url(urlpath, dirpath)
        f = tarfile.open(tar_gz_path)
        f.extractall(dirpath+basename) 
        f.close()
        os.remove(dirpath+os.path.basename(urlpath))

    for filepath in glob(dirpath+basename+'/*.txt'):
        srf = pd.read_csv(filepath, skiprows=4, 
                        engine='python', dtype=float,
                        index_col=False, 
                        delim_whitespace=True,
                        names=["Wavelength", "Response"])
        srf['Wavelength'] = srf['Wavelength'].apply(lambda x: 1./x*1e7).values # Convert wavelength in nm
        central_wav = round(srf['Wavelength'].mean())
        ds[central_wav] = xr.DataArray(srf['Response'].values, 
                                        coords={f'wav_{central_wav}':srf['Wavelength'].values})

    return ds


def combine_with_srf(srf:xr.Dataset, lut:xr.DataArray):
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