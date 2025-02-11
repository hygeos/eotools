from pathlib import Path
from typing import Optional
import pandas as pd
import xarray as xr
import numpy as np

from pyhdf.SD import SD
from core.tools import datetime
from core.fileutils import mdir
from eoread.download_legacy import download_url
from eotools.gaseous_absorption import get_absorption
from eotools.srf import integrate_srf
from core import env


class Gaseous_correction:
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
    dir_common : Optional[Path]
        Path to the `common` directory.

    Example
    -------
    Apply gaseous correction to `l1`

    >>> Gaseous_correction(l1).apply()
    """                 

    requires_anc = ['horizontal_wind', 'sea_level_pressure', 'total_column_ozone']

    def __init__(self,
                 ds: xr.Dataset,
                 srf: xr.Dataset,
                 input_var: str='rho_toa',
                 ouput_var: str='rho_gc',
                 spectral_dim: str='bands',
                 dir_common: Optional[Path]=None):

        self.ds = ds
        self.spectral_dim = spectral_dim
        self.bands = list(ds[spectral_dim].data)
        self.input_var = input_var
        self.output_var = ouput_var
        for b in self.bands:
            assert b in srf

        try:
            # image mode: single date for the whole image
            self.datetime = datetime(ds)
        except AttributeError:
            # extraction mode: per-pixel date
            self.datetime = None

        # Collect auxilary data
        if dir_common is None:
            dir_common = mdir(env.getdir("DIR_STATIC") / "common")
        self.no2_climatology = download_url(
            'https://docs.hygeos.com/s/5oWmND4xjmbCBtf/download/no2_climatology.hdf',
            dir_common, verbose=False,
        )
        self.no2_frac200m = download_url(
            'https://docs.hygeos.com/s/4tzqH25SwK9iGMw/download/trop_f_no2_200m.hdf',
            dir_common, verbose=False,
        )

        # load absorption rate for each gas
        k_oz_data  = get_absorption('o3', dirname=dir_common)
        k_no2_data = get_absorption('no2', dirname=dir_common)

        self.K_OZ = integrate_srf(srf, k_oz_data)
        self.K_NO2 = integrate_srf(srf, k_no2_data)

        # consistency checking: check that both wavc and ds.bands are sorted
        # wavc = integrate_srf(srf, lambda x: x)
        # assert (np.diff(ds.bands) > 0).all()
        # assert (np.diff(list(wavc.values())) > 0).all()

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

    def get_no2(self, latitude, longitude, datetime, flags):
        """
        returns no2_frac, no2_tropo, no2_strat at the pixels coordinates
        (latitude, longitude)
        """
        ok = ((flags) == 0)
        ok &= ~np.isnan(latitude) & ~np.isnan(longitude)

        # get month
        if not hasattr(datetime, "month"):
            mon = -1
            imon = pd.DatetimeIndex(datetime).month - 1
        else:
            mon = datetime.month
            imon = 0

        try:
            self.no2_tropo_data
        except:
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

    def run(self, bands, Rtoa, mus, muv,
            ozone, latitude, longitude,
            flags, 
            datetime):
        """
        Apply gaseous correction to Rtoa (ozone, NO2)

        ozone : total column in Dobson Unit (DU)
        """
        Rtoa_gc = np.zeros_like(Rtoa)+np.nan
        ok = ~np.isnan(latitude)   # FIXME:

        # TODO: nightpixels & invalid pixels

        air_mass = 1/muv + 1/mus

        # ozone correction
        for i, b in enumerate(bands):

            if not Rtoa.size:
                break 

            tauO3 = self.K_OZ[b] * ozone[ok] * 1e-3  # convert from DU to cm*atm

            # ozone transmittance
            trans_O3 = np.exp(-tauO3 * air_mass[ok])

            Rtoa_gc[ok, i] = Rtoa[ok, i]/trans_O3

        # NO2 correction
        no2_frac, no2_tropo, no2_strat = self.get_no2(latitude, longitude, datetime, flags)
        no2_tr200 = no2_frac * no2_tropo
        no2_tr200[no2_tr200 < 0] = 0

        for i, b in enumerate(bands):

            if not Rtoa.size:
                break 

            k_no2 = self.K_NO2[b]

            a_285 = k_no2 * (1.0 - 0.003*(285.0-294.0))
            a_225 = k_no2 * (1.0 - 0.003*(225.0-294.0))

            tau_to200 = a_285*no2_tr200 + a_225*no2_strat

            t_no2 = np.exp(-tau_to200[ok] * air_mass[ok])

            Rtoa_gc[ok, i] /= t_no2

        return Rtoa_gc.astype('float32')

    def apply(self):
        """Apply gaseous to current dataset `ds`

        This creates the variable `rho_gc` in the current dataset
        """        
        ds = self.ds
        date = datetime(ds)

        if ds.total_column_ozone.units in ['Kg.m-2', 'kg m**-2', 'kg.m-2']:
            total_ozone = ds.total_column_ozone / 2.1415e-5  # convert kg/m2 to DU
        else:
            total_ozone = ds.total_column_ozone
            assert ds.total_column_ozone.units in ['DU','Dobsons', 'Dobson']

        rho_toa = ds[self.input_var].chunk({self.spectral_dim: -1})
        ds[self.output_var] = xr.apply_ufunc(
            self.run,
            ds[self.spectral_dim],
            rho_toa,
            ds.mus,
            ds.muv,
            total_ozone,
            ds.latitude,
            ds.longitude,
            ds.flags,
            dask='parallelized',
            kwargs={'datetime': date},
            input_core_dims=[[self.spectral_dim], [self.spectral_dim], [], [], [], [], [], []],
            output_core_dims=[[self.spectral_dim]],
            output_dtypes=['float32'],
        )
