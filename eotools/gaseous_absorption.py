from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from core import env
from core.env import getdir
from core.files.cache import cache_dataset
from core.files.fileutils import mdir
from core.network.download import download_url
from core.pseudoinverse import pseudoinverse
from numpy import dot
from scipy.optimize import curve_fit

# Optional import for gatiab
try:
    from gatiab import Gatiab
    GATIAB_AVAILABLE = True
except ImportError:
    GATIAB_AVAILABLE = False
    Gatiab = None

from eotools.srf import filter_bands, get_SRF, rename


gas_list_gatiab = ["CH4", "CO2", "H2O", "N2", "N2O", "O2", "O3"]


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


def trans_func(x, a, n):
    return np.exp( -a*x**n )



def get_file_gatiab(gas: str, afglmod: str = 'afglus') -> Path:
    assert gas.upper() in gas_list_gatiab
    file_od_spectrum = (
        getdir("DIR_ABS_SPECTRUM_GATIAB", getdir("DIR_STATIC") / "absorption" / "gatiab")
        / afglmod
        / f"od_{gas.upper()}_afglus_ckdmip_idealized_solar_spectra.nc"
    )

    assert file_od_spectrum.exists()

    return file_od_spectrum


def get_abs_data(srf: xr.Dataset, gas: str) -> xr.Dataset:
    """
    Generate gaseous absorption transmission data for model fitting
    
    Parameters
    ----------
    srf : xr.Dataset
        Spectral Response Function dataset containing sensor band definitions.
        Each band should be a DataArray
    gas : str
        Gas species identifier. Must be one of the GATIAB-supported gases:
        ['CH4', 'CO2', 'H2O', 'N2', 'N2O', 'O2', 'O3']
        
    Returns
    -------
    xr.Dataset
    """
    if 'id' in srf:
        srf = srf.stack(bands=srf.id.dims)
        bnames = list(srf.id.values)
    else:
        bnames = list(srf)
    rsrf = [srf[b].values for b in bnames]
    srf_wvl = [srf[b][srf[b].dims[0]].values for b in bnames]

    # load gatiab file
    file_od_spectrum = get_file_gatiab(gas)
    ds = xr.open_dataset(file_od_spectrum)

    # Instantiate Gatiab object
    gt = Gatiab(ds)

    # Define air mass and gas content based on gas type with appropriate concentration
    # ranges
    air_mass = np.linspace(2.0, 6.0, 15)
    P0 = np.array(1013.25)
    if gas == 'H2O':
        U_H2O = np.linspace(0.5, 6.5, 15)
        ds_gas = gt.calc(U_H2O, air_mass, P0[None], srf_wvl, rsrf)
        U_units = 'g/cm²'
    elif gas == 'O3':
        U_O3 = np.linspace(100.0, 500.0, 15)
        ds_gas = gt.calc(U_O3, air_mass, P0[None], srf_wvl, rsrf)
        U_units = 'Dobson'
    else:
        # Ratio pressure/ground_pressure for other gases
        R_P_P0 = np.linspace(0.83909181, 1.0266535, 15)
        U_gas = R_P_P0 * gt.get_gas_content()
        ds_gas = gt.calc(U_gas, air_mass, P0[None], srf_wvl, rsrf)
        U_units = 'g/cm²'
    
    # Add nominal gas concentration
    ds_gas["U0"] = xr.DataArray(
        gt.get_gas_content(),
        attrs={"units": U_units, "description": "Nominal gas concentration."},
    )
    ds_gas["P0"] = xr.DataArray(
        P0,
        attrs={"units": "hPa", "description": "Nominal surface pressure."},
    )

    # Add metadata for gas type and concentration units
    ds_gas.attrs['gas'] = gas

    ds_gas = ds_gas.rename({'lambda': 'bands'}).assign_coords(bands=bnames)
    if 'id' in srf:
        ds_gas = ds_gas.assign_coords(bands=srf.bands)
    
    return ds_gas


def abs_data_fit(
    ds_gas: xr.Dataset, method: Literal["curve_fit", "pseudoinverse"] = "pseudoinverse"
) -> xr.Dataset:
    """
    Fit gaseous absorption coefficients for atmospheric transmission modeling.
    
    Fits the transmission function T(x) = exp(-a * x^n) where x = M * U
    (air mass times gas content) for each wavelength band.
    `ds_gas` is provided by `get_abs_data`.
    
    Two fitting methods are available:
    - "pseudoinverse": Uses matrix pseudoinverse on linearized form ln(-ln(T)) = ln(a) + n*ln(x)
    - "curve_fit": Uses scipy.optimize.curve_fit for non-linear optimization
    
    Parameters
    ----------
    ds_gas : xr.Dataset
        Dataset containing atmospheric transmission data
    method : {"curve_fit", "pseudoinverse"}, default "pseudoinverse"
        Fitting method to use for parameter estimation
        
    Returns
    -------
    xr.Dataset
        Dataset with fitted coefficients 'a' and 'n' for each wavelength band.
        Variables: and and n coefficients
        
    Notes
    -----
    The transmission model is: T(x) = exp(-a * x^n)
    where x = M * U (air mass x gas content)
    
    For the pseudoinverse method, the equation is linearized as:
    ln(-ln(T)) = ln(a) + n * ln(x)
    """
    
    nwav = len(ds_gas['bands'])
    a_coeffs = np.zeros(nwav) + np.nan
    n_coeffs = np.zeros(nwav) + np.nan
    
    for ilam in range(nwav):
        # calculate transmissions
        U = ds_gas.U.broadcast_like(ds_gas.trans).isel({"bands": ilam}).values.ravel()
        M = ds_gas.M.broadcast_like(ds_gas.trans).isel({"bands": ilam}).values.ravel()
        T = ds_gas.trans.isel({"bands": ilam}).values.ravel()
        U0 = ds_gas.U0.values
        
        if method == "curve_fit":
            # Fit: method 1
            p0 = np.array([np.log((T / (M * U))[-1]), 1.0])
            try:
                popt, pcov = curve_fit(
                    lambda x, a, n: trans_func(x, -a, n), M * U / U0, T, p0=p0
                )
            except RuntimeError:
                a, n = np.nan, np.nan
            finally:
                a = popt[0]
                n = popt[1]
        elif method == "pseudoinverse":

            if np.allclose(T, 1., rtol=1e-7):
                a = 0.
                n = 1.
            else:

                # Fit: method 2
                gamma = np.array([[1, np.log(M[i] * U[i] / U0)] for i in range(len(M))])
                X_ = dot(pseudoinverse(gamma), np.log(-np.log(T)))
                a = np.exp(X_[0])
                n = X_[1]

        a_coeffs[ilam] = a
        n_coeffs[ilam] = n
    
    # Create xarray Dataset
    coeffs = xr.Dataset(
        {
            'a': (['bands'], a_coeffs, {'description': 'Absorption coefficient', 'units': 'dimensionless'}),
            'n': (['bands'], n_coeffs, {'description': 'Exponent parameter', 'units': 'dimensionless'}),
            'U0': ds_gas.U0,
            'P0': ds_gas.P0,
            'U_units': ds_gas.U0.attrs['units'],
        },
        coords={'bands': ds_gas.bands},
        attrs={
            'fitting_method': method,
            'transmission_model': 'T(x) = exp(-a * x^n)',
            'model_variable': 'x = M * U / U0 (air_mass * gas_content / nominal_gas_content)',
            **ds_gas.attrs,
        }
    )
    
    coeffs = coeffs.unstack()
    if "bands" not in coeffs.dims:
        coeffs = coeffs.rename({coeffs.a.dims[0]: "bands"})
    return coeffs


def get_transmission_coeffs(
    platform_sensor: str | Tuple, create: bool = False
) -> xr.Dataset:
    """
    Calculate transmission coefficients for all gases, for a given sensor

    If `create`, allow the creation of the file containing the transmission
    coefficients.
    """
    version = 'v1'
    if isinstance(platform_sensor, tuple):
        psensor = ("_".join(platform_sensor)).upper()
    else:
        psensor = platform_sensor.upper()
    dir_gatiab = mdir(getdir('DIR_STATIC')/"absorption"/"gatiab")
    filename = dir_gatiab/f"abs_coeff_{psensor}_{version}.nc"

    if (not create) and (not filename.exists()):
        raise FileNotFoundError(filename)

    @cache_dataset(filename, attrs={'platform_sensor': psensor})
    def wrapped(platform_sensor_: str | Tuple):
        srf = get_SRF(platform_sensor_)
        srf = filter_bands(srf, 250, 2500)
        srf = rename(srf, band_ids='trim', thres_check=None)

        # generate all fit coefficients
        list_coeffs = [abs_data_fit(get_abs_data(srf, gas)) for gas in gas_list_gatiab]

        # concatenate all coefficients in a single array
        concatenated = xr.concat(
            list_coeffs, dim="gas", combine_attrs="drop_conflicts"
        ).assign_coords(gas=[x.gas for x in list_coeffs])

        return concatenated
    
    return wrapped(platform_sensor)