from pathlib import Path
from typing import Dict
import pandas as pd
import xarray as xr
import numpy as np

from pyhdf.SD import SD
from core.tools import datetime
from core.files.fileutils import mdir
from core.network.download import download_url
from eotools.gaseous_absorption import (
    get_absorption,
    transmission_model,
)
from eotools.srf import integrate_srf
from eotools.units import convert
from core import env
from core.process.blockwise import BlockProcessor, CompoundProcessor, Var


class Gaseous_correction(CompoundProcessor):
    """Gaseous correction module

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to which the gaseous correction is applied.
        The bands of `ds` shall be contained in `srf`.
    srf : xr.Dataset
        The sensor spectral response function (SRF).
    input_var: str
        Name of the input variable in `ds` (Top of atmosphere reflectance)
    ouput_var: str
        Name of the output variable in `ds`
    spectral_dim: str
        Name of the spectral dimension in input_var and output_var
    no2_correction: str
        Method for NO2 correction. Options: "legacy" (default, uses
        climatology-based NO2 correction) or any other value to skip.
    gas_correction: str
        Method for gas correction. Options: "o3_legacy" (default,
        legacy O3 correction), "ckdmip" (CKDMIP transmission coefficients).
    K_NO2 : Dict | None
        Explicit NO2 absorption coefficients per band. If ``None``,
        coefficients are calculated from the SRF or central wavelengths.
    K_OZ : Dict | None
        Explicit ozone absorption coefficients per band. If ``None``,
        coefficients are calculated from the SRF or central wavelengths.
    list_gases : list of str, optional
        Subset of gases to include in the CKDMIP correction (used when
        ``gas_correction='ckdmip'``).  Valid gas names are
        ``['CH4', 'CO2', 'H2O', 'N2', 'N2O', 'O2', 'O3']``.
        If ``None`` (default), all gases available in the transmission model
        are used.

    Example
    -------
    Apply gaseous correction to `l1`

    >>> Gaseous_correction(l1).apply()
    """                 
    def __init__(self,
                 ds: xr.Dataset,
                 srf: xr.Dataset | None = None,
                 input_var: str='rho_toa',
                 ouput_var: str='rho_gc',
                 spectral_dim: str='bands',
                 K_NO2: Dict | None = None,
                 K_OZ: Dict | None = None,
                 no2_correction: str = "legacy",
                 gas_correction: str = "o3_legacy",
                 list_gases: list | None = None,
                 dtype: str = "float32",
                 **kwargs
                 ):

        self.no2_correction = no2_correction
        self.gas_correction = gas_correction
        self.ds = ds
        self.spectral_dim = spectral_dim
        self.bands = list(ds[spectral_dim].data)
        self.input_var = input_var
        self.output_var = ouput_var
        self.dtype = dtype

        try:
            # image mode: single date for the whole image
            self.datetime = datetime(ds)
        except AttributeError:
            # extraction mode: per-pixel date
            self.datetime = None

        dir_common = mdir(env.getdir("DIR_STATIC") / "common")
        
        # Initialize sub classes
        processors: list[BlockProcessor] = [
            Init_rho_gc(input_var, dtype=dtype),
            Init_air_mass(ds),
        ]
        if no2_correction == "legacy":
            self.corr_NO2 = Gas_correction_NO2(ds, srf, dir_common, K_NO2, spectral_dim)
            processors.append(self.corr_NO2)
        if gas_correction == 'o3_legacy':
            self.corr_O3 = Gas_correction_O3(ds, srf, dir_common, K_OZ, spectral_dim)
            processors.append(self.corr_O3)
        elif gas_correction == 'ckdmip':
            assert srf is not None
            self.corr_ckdmip = Gas_correction_CKDMIP(srf, list_gases=list_gases)
            processors.append(self.corr_ckdmip)
        else:
            raise ValueError

        super().__init__(processors)


class Init_rho_gc(BlockProcessor):
    def __init__(self, input_var: str, dtype: str = "float32"):
        self.input_var = input_var
        self.dtype = dtype

    def input_vars(self) -> list[Var]:
        return [Var(self.input_var)]

    def created_vars(self) -> list[Var]:
        return [Var("rho_gc", dtype=self.dtype)]
    
    def auto_template(self) -> bool:
        return True

    def process_block(self, block: xr.Dataset):
        block["rho_gc"] = block[self.input_var].astype(self.dtype)


class Init_air_mass(BlockProcessor):
    def __init__(self, ds: xr.Dataset):
        self.activate = "air_mass" not in ds

    def input_vars(self) -> list[Var]:
        return [Var("mus"), Var("muv")] if self.activate else []

    def created_vars(self) -> list[Var]:
        return [Var("air_mass")]
    
    def auto_template(self) -> bool:
        return True

    def process_block(self, block: xr.Dataset):
        block['air_mass'] = 1/block.muv + 1/block.mus



class Gas_correction_O3(BlockProcessor):
    """
    Ozone absorption correction module
    """

    def __init__(
        self,
        ds: xr.Dataset,
        srf: xr.Dataset | None,
        dir_common: Path,
        K_OZ: Dict | None = None,
        spectral_dim: str='bands',
    ):
        self.spectral_dim = spectral_dim
        self.bands = list(ds[spectral_dim].data)

        if K_OZ is not None:
            # K_OZ is provided explicitly
            self.K_OZ = xr.Dataset()
            for k, v in K_OZ.items():
                self.K_OZ[k] = v
        else:
            # K_OZ calculated from SRF

            # load absorption rate for each gas
            k_oz_data  = get_absorption('o3', dirname=dir_common)

            if srf is not None: # K_OZ and K_NO2 are calculated from srf
                for b in self.bands:
                    assert b in srf

                self.K_OZ = integrate_srf(srf, k_oz_data, resample='x')

                # consistency checking: check that both wavc and ds.bands are sorted
                # wavc = integrate_srf(srf, lambda x: x)
                # assert (np.diff(ds.bands) > 0).all()
                # assert (np.diff(list(wavc.values())) > 0).all()
        
            else: # K_OZ is calculated from central wavelengths
                self.K_OZ = xr.Dataset()
                for i, k in enumerate(k_oz_data.interp(wav=ds.cwav).values):
                    self.K_OZ[self.bands[i]] = k
    
    def run(self, bands, Rtoa_gc, ok, air_mass,
            ozone):
        """
        Apply O3 transmission

        ozone : total column in Dobson Unit (DU)
        """

        # ozone correction
        for i, b in enumerate(bands):

            if not Rtoa_gc.size:
                break 

            tauO3 = self.K_OZ[b].values * ozone[ok] * 1e-3  # convert from DU to cm*atm

            # ozone transmittance
            trans_O3 = np.exp(-tauO3 * air_mass[ok])

            Rtoa_gc[ok, i] /= trans_O3

        return Rtoa_gc.astype('float32')
    
    def input_vars(self) -> list[Var]:
        return [
            Var("air_mass"),
            Var("total_column_ozone"),
            Var("latitude"),
            Var("longitude"),
        ]

    def modified_vars(self) -> list[Var]:
        return [Var('rho_gc')]
    
    def process_block(self, block: xr.Dataset):
        ok = ~np.isnan(block.latitude.data)
        ozone_dobson = convert(block.total_column_ozone, 'Dobson').data
        self.run(
            block[self.spectral_dim].data,
            block.rho_gc.transpose(..., 'bands').data,
            ok,
            block.air_mass.data,
            ozone_dobson
            )


class Gas_correction_NO2(BlockProcessor):
    def __init__(
        self,
        ds: xr.Dataset,
        srf: xr.Dataset | None,
        dir_common: Path,
        K_NO2: Dict | None = None,
        spectral_dim: str='bands',
    ):
        self.spectral_dim = spectral_dim
        self.bands = list(ds[spectral_dim].data)
        self.datetime = datetime(ds)

        # Collect auxilary data
        self.no2_climatology = download_url(
            'https://github.com/hygeos/eotools/releases/download/root/no2_climatology.hdf',
            dir_common, verbose=False,
        )
        self.no2_frac200m = download_url(
            'https://github.com/hygeos/eotools/releases/download/root/trop_f_no2_200m.hdf',
            dir_common, verbose=False,
        )

        if K_NO2 is not None:
            # K_NO2 is provided explicitly
            self.K_NO2 = xr.Dataset()
            for k, v in K_NO2.items():
                self.K_NO2[k] = v
        else:
            # load absorption rate for each gas
            k_no2_data = get_absorption('no2', dirname=dir_common)

            if srf is not None:
                for b in self.bands:
                    assert b in srf

                self.K_NO2 = integrate_srf(srf, k_no2_data, resample='x')

                # consistency checking: check that both wavc and ds.bands are sorted
                # wavc = integrate_srf(srf, lambda x: x)
                # assert (np.diff(ds.bands) > 0).all()
                # assert (np.diff(list(wavc.values())) > 0).all()
        
            else: # K_OZ and K_NO2 are calculated from central wavelengths
                self.K_NO2 = xr.Dataset()
                for i, k in enumerate(k_no2_data.interp(wav=ds.cwav).values):
                    self.K_NO2[self.bands[i]] = k

    def run(
        self,
        bands,
        Rtoa_gc,
        ok,
        air_mass,
        latitude,
        longitude,
        datetime,
    ):
        """
        Apply NO2 transmission
        """

        # NO2 correction
        no2_frac, no2_tropo, no2_strat = self.get_no2(latitude, longitude, datetime)
        no2_tr200 = no2_frac * no2_tropo
        no2_tr200[no2_tr200 < 0] = 0

        for i, b in enumerate(bands):

            if not Rtoa_gc.size:
                break 

            k_no2 = self.K_NO2[b].values

            a_285 = k_no2 * (1.0 - 0.003*(285.0-294.0))
            a_225 = k_no2 * (1.0 - 0.003*(225.0-294.0))

            tau_to200 = a_285*no2_tr200 + a_225*no2_strat

            t_no2 = np.exp(-tau_to200[ok] * air_mass[ok])

            Rtoa_gc[ok, i] /= t_no2
        

    def read_no2_data(self, month):
        '''
        read no2 data from month (1..12) or for all months if month < 0

        shape of arrays is (month, lat, lon)
        '''
        hdf1 = SD(str(self.no2_climatology))
        hdf2 = SD(str(self.no2_frac200m))

        if month < 0:
            months = range(1, 13)
            nmo = 12
        else:
            months = [month]
            nmo = 1

        self.no2_total_data = np.zeros((nmo,720,1440), dtype='float32')
        self.no2_tropo_data = np.zeros((nmo,720,1440), dtype='float32')
        self.no2_frac200m_data = np.zeros((90,180), dtype='float32')

        for i, m in enumerate(months):
            # read total and tropospheric no2 data
            self.no2_total_data[i,:,:] = hdf1.select('tot_no2_{:02d}'.format(m)).get()

            self.no2_tropo_data[i,:,:] = hdf1.select('trop_no2_{:02d}'.format(m)).get()

        # read fraction of tropospheric NO2 above 200mn
        self.no2_frac200m_data[:,:] = hdf2.select('f_no2_200m').get()

        hdf1.end()
        hdf2.end()

    def get_no2(self, latitude, longitude, datetime):
        """
        returns no2_frac, no2_tropo, no2_strat at the pixels coordinates
        (latitude, longitude)
        """
        ok = ~np.isnan(latitude) & ~np.isnan(longitude)

        # get month
        if not hasattr(datetime, "month"):
            mon = -1
            imon = pd.DatetimeIndex(datetime).month - 1
        else:
            mon = datetime.month
            imon = 0

        try:
            self.no2_tropo_data
        except Exception:
            self.read_no2_data(mon)

        # coordinates of current block in 1440x720 grid
        ilat = (4*(90 - latitude)).astype('int')
        ilon = (4*longitude).astype('int')
        ilon[ilon<0] += 4*360
        ilat[~ok] = 0
        ilon[~ok] = 0

        no2_tropo = self.no2_tropo_data[imon,ilat,ilon]*1e15
        no2_strat = (self.no2_total_data[imon,ilat,ilon]
                     - self.no2_tropo_data[imon,ilat,ilon])*1e15

        # coordinates of current block in 90x180 grid
        ilat = (0.5*(90 - latitude)).astype('int')
        ilon = (0.5*longitude).astype('int')
        ilon[ilon<0] += 180
        ilat[~ok] = 0
        ilon[~ok] = 0
        no2_frac = self.no2_frac200m_data[ilat,ilon]

        return no2_frac, no2_tropo, no2_strat

    def input_vars(self) -> list[Var]:
        return [
            Var('latitude'),
            Var('longitude'),
            Var('air_mass'),
        ]
        
    def modified_vars(self) -> list[Var]:
        return [Var('rho_gc')]
    
    def process_block(self, block: xr.Dataset):
        ok = ~np.isnan(block.latitude.data)
        self.run(
            block[self.spectral_dim].data,
            block.rho_gc.transpose(..., 'bands').data,
            ok,
            block.air_mass.data,
            block.latitude.data,
            block.longitude.data,
            self.datetime,
        )


class Gas_correction_CKDMIP(BlockProcessor):
    """CKDMIP-based gaseous correction using transmission model coefficients.

    Computes band-averaged transmission coefficients by integrating over the
    sensor spectral response function (SRF).  Supports correction for multiple
    absorbing gases (O3, H2O, CH4, CO2, N2, N2O, etc.) using a parametric
    model ``T = Teq ** (x ** n)`` where ``x`` is the normalized column amount.

    Parameters
    ----------
    srf : xr.Dataset
        The sensor spectral response function.  Must contain all bands
        present in the input dataset.
    thres_correction : float, optional
        Minimum transmission threshold.  Transmission values below this
        threshold are replaced with ``NaN`` to avoid extreme reflectance
        corrections.  Default is 0.5.
    list_gases : list of str, optional
        Gases to include in the correction.  Default is all available gases
        in the transmission model: ``['CH4', 'CO2', 'H2O', 'N2', 'N2O', 'O2', 'O3']``.
    """

    def __init__(self, srf: xr.Dataset, thres_correction: float = 0.5, list_gases: list | None = None):
        # Determine the transmission model by integrating over the SRF for each gas
        self.tmod = transmission_model(srf)
        self.thres_correction = thres_correction

        if list_gases is None:
            self.list_gases = list(self.tmod)
        else:
            self.list_gases = list_gases
    
    def input_vars(self) -> list[Var]:
        return [
            Var("latitude"),
            Var("longitude"),
            Var("air_mass"),
            Var("total_column_water_vapour"),
            Var("total_column_ozone"),
            Var("sea_level_pressure"),
        ]

    def modified_vars(self) -> list[Var]:
        return [Var('rho_gc')]
        
    def process_block(self, block: xr.Dataset):
        """Apply CKDMIP gaseous correction to a single block.

        Computes the combined transmission from all gases (O3, H2O, etc.)
        using log-space accumulation for numerical stability:
            prod(Teq^(x^n)) = exp(sum(x^n * log(Teq)))
        then divides the reflectance by the total gas transmission.
        """
        Rtoa_gc = block.rho_gc.transpose(..., 'bands').values
        air_mass = block.air_mass.values

        ozone = convert(block.total_column_ozone, 'Dobson').values
        sea_surface_pressure = convert(block.sea_level_pressure, 'hPa').values
        tcwv = convert(block.total_column_water_vapour, 'g/cm²').values

        ok = ~np.isnan(block.latitude.values)

        n_bands = Rtoa_gc.shape[-1]
        n_pixels = ok.sum()
        log_T_total = np.zeros((n_pixels, n_bands), dtype=np.float64)

        for gas in self.list_gases:
            U0 = self.tmod[gas]['U0']
            P0 = self.tmod[gas]['P0']
            if gas == 'O3':
                assert U0.units == "Dobson"
                x = air_mass[ok] * ozone[ok] / U0.values
            elif gas == 'H2O':
                assert U0.units == "g/cm²"
                x = air_mass[ok] * tcwv[ok] / U0.values
            else:
                assert P0.units == 'hPa'
                x = air_mass[ok] * sea_surface_pressure[ok] / P0.values

            n = np.nan_to_num(self.tmod[gas].n.values, nan=1.0)
            log_Teq = np.log(self.tmod[gas].Teq.values)

            # Skip bands with near-unity transmission (log_Teq ≈ 0)
            mask = np.abs(log_Teq) > 1e-6
            if mask.any():
                log_T_total[:, mask] += (x[:, None] ** n[None, mask]) * log_Teq[None, mask]

        T = np.exp(log_T_total)
        T[T < self.thres_correction] = np.nan

        Rtoa_gc[ok, :] /= T

