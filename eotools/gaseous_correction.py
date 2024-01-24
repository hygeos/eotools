from datetime import datetime as dt
from os import makedirs
import pandas as pd
import xarray as xr
import numpy as np

from pyhdf.SD import SD
from pathlib import Path
from eoread.eo import datetime, init_geometry
from eotools.gaseous_absorption import get_climate_data, get_absorption, combine_with_srf
from eotools.srf import get_SRF

from eoread.ancillary.era5 import ERA5


class Gaseous_correction:
    '''
    Gaseous correction module

    Ex: Gaseous_correction(l1).apply()
    '''

    requires_anc = ['u_wind_at_10m', 'sea_level_pressure', 'total_column_ozone']

    def __init__(self, ds, srf=None):
        self.ds = ds
        self.bands = list(ds.bands.data)

        if 'datetime' in ds.attrs:
            # image mode: single date for the whole image
            self.datetime = datetime(ds)
        else:
            # extraction mode: per-pixel date
            self.datetime = None

        # Collect auxilary data
        dir_base = Path(__file__).parent.parent
        dir_common = dir_base/'auxdata'/'common'
        get_climate_data(dir_common)
        self.no2_climatology = dir_common/'no2_climatology.hdf'
        self.no2_frac200m = dir_common/'trop_f_no2_200m.hdf'
        assert self.no2_climatology.exists()
        assert self.no2_frac200m.exists()
            
        # Load srf for each channel
        if srf is None:
            srf = get_SRF(l1_ds=ds)
        else:
            assert isinstance(srf,xr.Dataset)

        # load absorption rate for each gas
        k_oz_data  = get_absorption('o3', dir_common)
        k_no2_data = get_absorption('no2', dir_common)
        
        # calculates the correction coefficients for each gas
        for name_k,k in zip(['K_oz', 'K_no2'],[k_oz_data, k_no2_data]):
            dic_k = combine_with_srf(srf, k)
            bands_idx = [(np.abs(np.array(self.bands) - k)).argmin() for k in dic_k.keys()]
            dic_k = {self.bands[bands_idx[k]]: val for k,val in enumerate(dic_k.values())}

            # stores calculated variables in the dataset
            if name_k == 'K_oz' : 
                self.K_OZ  = dic_k
            if name_k == 'K_no2': 
                self.K_NO2 = dic_k
        
        # Collect ancillary data
        self.add_ancillary(self.ds)

        # Initialize angles geometry
        init_geometry(self.ds)
    

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
        ok = (flags) == 0

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
    

    def run(self, Rtoa, mus, muv,
            ozone, latitude, longitude,
            flags, 
            datetime):
        """
        Apply gaseous correction to Rtoa (ozone, NO2)

        ozone : total column in Dobson Unit (DU)
        """
        Rtoa_gc = np.zeros_like(Rtoa)+np.NaN
        ok = ~np.isnan(latitude)   # FIXME:

        # TODO: nightpixels
        # TODO: invalid pixelshttps://docs.hygeos.com/s/M7iK4eX4CbpYKj8/download/LUT.hdf

        air_mass = 1/muv + 1/mus

        # ozone correction
        for i, b in enumerate(self.K_OZ.keys()):

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

        for i, b in enumerate(self.K_OZ.keys()):

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
        ds = self.ds

        if ds.total_ozone.units in ['Kg.m-2', 'kg m**-2']:
            total_ozone = ds.total_ozone / 2.1415e-5  # convert kg/m2 to DU
        else:
            total_ozone = ds.total_ozone
            assert ds.total_ozone.units in ['DU','Dobsons']
        
        date = dt.strptime(ds.datetime, '%Y-%m-%dT%H:%M:%S')
        Rtoa = ds.Rtoa.chunk(dict(bands=-1))
        ds['rho_gc'] = xr.apply_ufunc(
            self.run,
            Rtoa,
            ds.mus,
            ds.muv,
            total_ozone,
            ds.latitude,
            ds.longitude,
            ds.flags,
            dask='parallelized',
            kwargs={'datetime': date},
            input_core_dims=[['bands'], [], [], [], [], [], []],
            output_core_dims=[['bands']],
            output_dtypes=['float32'],
        )

        return ds['rho_gc']