from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from core import env
from core.env import getdir
from core.files.cache import cache_dataset
from core.files.fileutils import filegen, mdir
from core.network.download import download_url
from core.pseudoinverse import pseudoinverse
from numpy import dot
from scipy.optimize import curve_fit
from core.tools import only

# Optional import for gatiab
try:
    from gatiab import Gatiab
    GATIAB_AVAILABLE = True
except ImportError:
    GATIAB_AVAILABLE = False
    Gatiab = None

from eotools.srf import filter_bands, get_SRF, get_bands, rename


gas_list_gatiab = ["CH4", "CO2", "H2O", "N2", "N2O", "O2", "O3"]


def get_climate_data(dirname: Union[str, Path]) -> None:
    """Download NO₂ climatology climate data files from the Hygeos server.

    Fetches two HDF files:
    - ``no2_climatology.hdf`` — NO₂ climatology data
    - ``trop_f_no2_200m.hdf`` — tropospheric NO₂ factor at 200 m resolution

    Parameters
    ----------
    dirname : str or Path
        Directory where the files will be downloaded.
    """
    download_url('https://docs.hygeos.com/s/5oWmND4xjmbCBtf/download/no2_climatology.hdf',dirname)
    download_url('https://docs.hygeos.com/s/4tzqH25SwK9iGMw/download/trop_f_no2_200m.hdf',dirname)


def get_absorption(gaz: str, dirname: Optional[Union[str, Path]]=None):
    """Read absorption cross-section data for a given gas.

    Downloads (if necessary) and parses a tabular text file that contains
    wavelength-dependent absorption values for the requested gas species.
    The metadata (URL, skip-rows, etc.) is looked up in *absorption_MTL.csv*.

    Parameters
    ----------
    gaz : str
        Gas species identifier (e.g. "O3", "NO2").  Must match a
        ``gaz`` entry in the absorption metadata file.
    dirname : str, Path, or None, optional
        Directory in which to cache the downloaded file.  Defaults to
        DIR_STATIC/common.

    Returns
    -------
    xr.DataArray
        1-D array of absorption values with a ``wav`` coordinate (nm).
    """
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
    """Return the path to a GATIAB optical-depth spectrum NetCDF file.

    Parameters
    ----------
    gas : str
        Gas species identifier. Must be one of ``gas_list_gatiab``:
        ['CH4', 'CO2', 'H2O', 'N2', 'N2O', 'O2', 'O3'].
    afglmod : str, optional
        Atmospheric profile model directory name, by default 'afglus'.

    Returns
    -------
    Path
        Absolute path to the optical-depth spectrum file.
    """
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

    Note: this function can be deprecated
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


def get_abs_data_hires(gas: str, afglmod: str = 'afglus') -> xr.Dataset:
    """
    Generate high-resolution gaseous absorption transmission at unit air mass
    and nominal gas content.

    Returns a gaseous transmission value per wavelength, computed for:
    - Air mass M = 1.0 (zenith path)
    - Nominal gas content U = U0 (from GATIAB)

    Parameters
    ----------
    gas : str
        Gas species identifier. Must be one of the GATIAB-supported gases:
        ['CH4', 'CO2', 'H2O', 'N2', 'N2O', 'O2', 'O3']
    afglmod : str, optional
        Atmospheric model to use, by default 'afglus'

    Returns
    -------
    xr.Dataset
        Dataset with a single 1-D `T1` variable over the `wav` dimension
        (wavelength in nm), plus scalar `U0` and `P0`.
    """
    from scipy import constants as scipy_constants
    from scipy.integrate import simpson
    from scipy.interpolate import interp1d
    from gatiab.gatiab import molar_mass as gatiab_molar_mass

    file_od_spectrum = get_file_gatiab(gas, afglmod=afglmod)
    ds = xr.open_dataset(file_od_spectrum)

    gt = Gatiab(ds)

    wavenum = gt.wavenumber  # cm^-1, shape (nwav,)
    P0 = np.float64(1013.25)

    # --- Nominal gas content ---
    U0 = gt.get_gas_content()
    if gas == 'H2O':
        U_units = 'g/cm²'
    elif gas == 'O3':
        U_units = 'Dobson'
    else:
        U_units = 'g/cm²'

    # --- Build pressure/height grid at ground pressure P0 (same as gt.calc) ---
    dens_gas_FPhl = interp1d(gt.P_hl, gt.dens_gas_hl, bounds_error=False, fill_value='extrapolate')

    mask = gt.P_hl <= P0 * 1e2
    z_atm_P0 = gt.z_atm[mask]
    dens_P0 = dens_gas_FPhl(gt.P_hl[mask]).astype(np.float64)
    denom = simpson(y=dens_P0, x=-z_atm_P0) * 1e5

    # --- Total column optical depth per wavenumber ---
    od_col = ds['optical_depth'].values.astype(np.float64).sum(axis=0)  # (nwav,)

    # --- Gas column scaling for nominal content ---
    if gas == 'O3':
        fac = 2.6867e16 * U0 / denom
    else:
        fac = U0 / gatiab_molar_mass[gas.lower()] * scipy_constants.Avogadro / denom

    # --- Transmission at M=1, U=U0: T(w) = exp(-fac * od_col[w]) ---
    trans = np.exp(-fac * od_col).astype(np.float32)  # (nwav,)

    # --- Build output dataset ---
    wav_nm = 1e7 / wavenum  # convert wavenumber (cm^-1) to wavelength (nm)

    ds_gas = xr.Dataset({
        'T1': xr.DataArray(
            trans,
            dims=['wav'],
            coords={'wav': wav_nm.astype('float32')},
            attrs={'description': 'Gaseous transmission at unit air mass and nominal gas content'},
        ),
    })
    ds_gas['wav'].attrs = {'units': 'nm', 'long_name': 'wavelength'}

    ds_gas['U0'] = xr.DataArray(
        U0,
        attrs={"units": U_units, "description": "Nominal gas concentration."},
    )
    ds_gas['P0'] = xr.DataArray(
        P0,
        attrs={"units": "hPa", "description": "Nominal surface pressure."},
    )
    ds_gas.attrs['gas'] = gas

    # Reverse so wavelength is increasing
    ds_gas = ds_gas.isel(wav=slice(None, None, -1))

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
    # TODO: deprecate this function
    
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
            'transmission_model': 'T(x) = T0^x',
            'model_variable': 'x = M * U / U0 (air_mass * gas_content / nominal_gas_content)',
            **ds_gas.attrs,
        }
    )
    
    coeffs = coeffs.unstack()
    if "bands" not in coeffs.dims:
        coeffs = coeffs.rename({coeffs.a.dims[0]: "bands"})
    return coeffs


def create_gaseous_abs(filename: Path) -> None:
    """Create a high-resolution gaseous absorption model DataTree and write to NetCDF.

    For each gas in ``gas_list_gatiab``, computes high-resolution transmission
    data at unit air mass and nominal gas content via :func:`get_abs_data_hires`,
    then packs the coefficients into ``uint16`` and writes a single NetCDF file.

    Parameters
    ----------
    filename : Path
        Output NetCDF file path.  The parent directory will be created
        if it does not exist.

    Raises
    ------
    ValueError
        If wavelength coordinates are inconsistent across gas datasets.
    """
    dt = None
    wav_root = None

    for gas in gas_list_gatiab:
        ds_gas = get_abs_data_hires(gas)

        if wav_root is None:
            wav_root = ds_gas["wav"].copy(deep=True)
            dt = xr.DataTree(dataset=xr.Dataset(coords={"wav": wav_root}))
        elif not ds_gas["wav"].identical(wav_root):
            raise ValueError(f"Inconsistent wav coordinate for gas {gas}")

        assert dt is not None
        dt[gas] = ds_gas.drop_vars("wav")

    assert dt is not None

    write_coeffs_uint16_netcdf(dt, filename=filename)


def get_gaseous_abs() -> xr.DataTree:
    """Load a DataTree of high-resolution absorption models for gases.

    The root dataset stores the common `wav` coordinate, and each gas is added
    as a child node containing fitted coefficients (without duplicating `wav`).

    Returns
    -------
    xr.DataTree
        A tree with one child dataset per gas in `gas_list_gatiab`.
        Each Dataset holds a variable T1.
    """
    dirname = env.getdir('DIR_STATIC')/'absorption'

    filename = download_url(
        "https://github.com/hygeos/eotools/releases/download/root/gaseous_absorption_model_gatiab_ckdmip_v1.nc",
        dirname,
    )

    # Here is the code that was used to create this file
    # filename = dirname/'gaseous_absorption_model_gatiab_ckdmip_v1.nc'
    # filegen(if_exists="skip")(create_gaseous_abs)(filename)

    return xr.open_datatree(filename)


def transmission_model_single(
    T: xr.DataArray, srf_band: xr.DataArray, x0: float = 2.0
) -> tuple:
    """
    Determine transmission model (Teq, n) for a given srf (for a single band)
        T(lam) = Teq^(x^n) where x = M*U/U0
    
    The model is fitted by evaluating the integrated transmission and its derivative at x = x0.
    
    Parameters
    ----------
    T : xr.DataArray
        High-resolution transmission spectrum T(lambda)
    srf_band : xr.DataArray
        Spectral response function for the band
    x0 : float, default 2.0
        Reference value of x = M*U/U0 at which to evaluate the derivative for fitting n
    
    Returns
    -------
    tuple
        (Teq, n) where Teq is the equivalent transmission at x=1, and n is the exponent
    """
    # Resample srf to T
    srf_bandname = only(srf_band.dims)
    srf_values = srf_band.interp({srf_bandname: T.wav})
    srf_values = srf_values.fillna(0.)

    A = np.trapezoid(srf_values * (T**x0), x=T.wav)
    B = np.trapezoid((srf_values * (T**x0) * np.log(T)).fillna(0.), x=T.wav)  # Fillna(0) because T.ln(T) -> 0 when T->0
    C = np.trapezoid(srf_values, x=T.wav)

    Tbar = A/C
    Tbarp = B/C

    n = Tbarp * x0 / (Tbar * np.log(Tbar))
    Teq = Tbar**(x0**-n)
    
    return (Teq, n)

def transmission_model(srf: xr.Dataset, x0: float = 5.) -> xr.DataTree:
    """
    Determine transmission model (Teq, n) for each gas , for a given srf
        T(lam) = Teq^(x^n) where x = M*U/U0

    The model is fitted by evaluating the integrated transmission and its derivative at x = x0.

    Parameters
    ----------
    srf : xr.Dataset
        Spectral response functions for all bands
    x0 : float, default 5.0
        Reference value of x = M*U/U0 at which to evaluate the derivative for fitting n

    Returns
    -------
    xr.DataTree
        DataTree with root having 'bands' coordinate, and each gas node containing
        'Teq' and 'n' variables for each band
    """
    gabs = get_gaseous_abs()

    list_bands = get_bands(srf)
    
    dt = xr.DataTree(dataset=xr.Dataset(coords={"bands": list_bands}))
    
    for gas in gas_list_gatiab:
        T = gabs[gas].T1
        teq_list = []
        n_list = []
        
        for band in list_bands:
            srf_band = srf[band]
            Teq, n = transmission_model_single(T, srf_band, x0=x0)
            teq_list.append(Teq)
            n_list.append(n)
        
        # Create xarray Dataset for this gas
        ds_gas = xr.Dataset(
            {
                'Teq': (['bands'], teq_list),
                'n': (['bands'], n_list),
            },
            coords={'bands': list_bands},
        )
        
        dt[gas] = ds_gas
    
    return dt


def transmission_model_eval(mod: xr.Dataset, x: xr.DataArray) -> xr.DataArray:
    """
    Evaluate the transmission model returned by transmission_model_srf, at provided `x`
    values.
    """
    Teq = mod.Teq
    n = mod.n
    return Teq ** (x ** n)


def write_coeffs_uint16_netcdf(
    coeffs: xr.DataTree,
    filename: Union[str, Path],
    *,
    engine: str = "h5netcdf",
) -> Path:
    """
    Pack numeric coefficient variables into uint16 and write to NetCDF.

    Numeric variables are stored with dtype=uint16 using scale_factor and
    add_offset so values are approximately preserved on read. Non-numeric
    variables are written without packing.

    Parameters
    ----------
    coeffs : xr.DataTree
        Coefficient tree to write. Packing is applied independently to each
        node dataset.
    filename : str | Path
        Output NetCDF path.
    engine : str, default "h5netcdf"
        NetCDF backend engine passed to xarray.

    Returns
    -------
    Path
        Path to written NetCDF file.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    int_dtype_max = np.iinfo(np.uint16).max
    fill_value = np.uint16(int_dtype_max)
    packed_max = np.float64(int_dtype_max - 1)

    def _build_encoding(ds: xr.Dataset) -> dict[str, dict]:
        encoding: dict[str, dict] = {}
        for var_name, data_array in ds.data_vars.items():
            if not np.issubdtype(data_array.dtype, np.number):
                continue

            values = np.asarray(data_array.values, dtype=np.float64)
            valid = np.isfinite(values)
            assert valid.all()
            assert (~np.isnan(values)).all()
            if not np.any(valid):
                # Keep a valid packing setup even if all values are missing.
                scale_factor = np.float64(1.0)
                add_offset = np.float64(0.0)
            else:
                vmin = np.nanmin(values)  # FIXME
                vmax = np.nanmax(values)  # FIXME
                assert np.isfinite(vmin)  # DEBUG
                assert np.isfinite(vmax)  # DEBUG
                if np.isclose(vmax, vmin):
                    scale_factor = np.float64(1.0)
                    add_offset = np.float64(vmin)
                else:
                    scale_factor = np.float64((vmax - vmin) / packed_max)
                    add_offset = np.float64(vmin)

            encoding[var_name] = {
                "dtype": "uint16",
                "scale_factor": scale_factor,
                "add_offset": add_offset,
                "_FillValue": fill_value,
                "zlib": True,
                "complevel": 9,
                "shuffle": True,
            }
        return encoding

    encoding = {
        node.path: _build_encoding(node.ds)
        for _, node in coeffs.subtree_with_keys
        if node.ds is not None
    }

    filegen()(coeffs.to_netcdf)(filename, encoding=encoding, engine=engine)
    return filename


def get_transmission_coeffs(
    platform_sensor: str | Tuple, create: bool = False
) -> xr.Dataset:
    """
    Calculate transmission coefficients for all gases, for a given sensor

    If `create`, allow the creation of the file containing the transmission
    coefficients.

    Note: this function can be deprecated in favour of `transmission_model`
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