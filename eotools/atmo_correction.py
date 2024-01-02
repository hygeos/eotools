from warnings import warn
from datetime import datetime as dt
from os import makedirs
import pandas as pd
import xarray as xr
import numpy as np

from lib.eoread.eo import datetime, init_geometry
from lib.eoread.download import FTP, get_auth_ftp, ftp_download
from pyhdf.SD import SD
from pathlib import Path
from eotools.sensor_srf import get_climate_data, get_SRF, get_absorption, combine_with_srf
from eotools.utils_odr import ODR

from lib.eoread.ancillary.era5 import ERA5


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
            srf = get_SRF(ds.sensor, ds.platform)
        else:
            assert isinstance(srf,xr.Dataset)

        # load absorption rate for each gas
        k_oz_data  = get_absorption('o3', dir_common)
        k_no2_data = get_absorption('no2', dir_common)
        
        # calculates the correction coefficients for each gas
        for name_k,k in zip(['K_oz', 'K_no2'],[k_oz_data, k_no2_data]):
            dic_k = combine_with_srf(srf, k)

            # stores calculated variables in the dataset
            if name_k == 'K_oz' : 
                self.K_OZ  = dic_k
            if name_k == 'K_no2': 
                self.K_NO2 = dic_k
        
        # Collect ancillary data
        self.add_ancillary(self.ds)
    

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
    

    def add_ancillary(self, ds):
        dir_root = Path(__file__).parents[1]
        makedirs(dir_root/'ancillary', exist_ok=True)

        var_to_get = ['total_column_ozone']
        varnames = ['total_ozone']

        # var_to_get = ['u_wind_at_10m', 'sea_level_pressure', 'total_column_ozone']
        # varnames = ['horizontal_wind', 'sea_level_pressure', 'total_ozone']

        era5 = ERA5(model = ERA5.models.reanalysis_single_level,
                    directory = dir_root/'ancillary')
        date = dt.strptime(ds.datetime,'%Y-%m-%dT%H:%M:%S')
        anc = era5.get(variables=var_to_get, dt=date)
        for varname in zip(varnames, var_to_get):
            ds[varname[0]] = anc[varname[1]].interp(
                latitude=ds.latitude,
                longitude=ds.longitude,
                ).drop(['latitude', 'longitude'])
    
    
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

        #
        # ozone correction
        #
        ozone_warn = (ozone[ok] < 50) | (ozone[ok] > 1000)
        if ozone_warn.any():
            warn('ozone is assumed in DU ({})'.format(ozone[ok][ozone_warn]))

        for i, b in enumerate(self.bands):

            if not Rtoa.size:
                break 

            tauO3 = self.K_OZ[b] * ozone[ok] * 1e-3  # convert from DU to cm*atm

            # ozone transmittance
            trans_O3 = np.exp(-tauO3 * air_mass[ok])

            Rtoa_gc[ok, i] = Rtoa[ok, i]/trans_O3

        #
        # NO2 correction
        #
        no2_frac, no2_tropo, no2_strat = self.get_no2(latitude, longitude, datetime, flags)
        no2_tr200 = no2_frac * no2_tropo
        no2_tr200[no2_tr200 < 0] = 0

        for i, b in enumerate(self.bands):

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
            assert ds.total_ozone.units in ['DU']
        
        ds['rho_gc'] = xr.apply_ufunc(
            self.run,
            ds.rho_toa,
            ds.mus,
            ds.muv,
            total_ozone,
            ds.latitude,
            ds.longitude,
            ds.flags,
            dask='parallelized',
            kwargs={'datetime': ds.start_time + (ds.end_time - ds.start_time)/2},
            input_core_dims=[['bands'], [], [], [], [], [], []],
            output_core_dims=[['bands']],
            output_dtypes=['float32'],
        )

        return ds['rho_gc']


class Rayleigh_correction:
    '''
    Rayleigh correction module

    Ex: Rayleigh_correction(l1).apply()
    '''

    requires_anc = []

    def __init__(self, ds, odr=None, srf=None):
        self.ds = ds
        self.bands = list(ds.bands.data)

        lut_path = Path('auxdata/luts/LUT_Rayleigh_PP.nc')
        if not lut_path.exists():
            ftp = FTP(**get_auth_ftp('GEO-OC'))
            ftp_download(ftp, lut_path, lut_path.parent)
        assert lut_path.exists()
        lut = xr.open_dataset(lut_path, chunks=-1)
        
        # set standard convention, raa = saa - vaa
        self.lut = lut.assign_coords(raa=180-lut.raa)    

        if 'datetime' in ds.attrs:
            # image mode: single date for the whole image
            self.datetime = datetime(ds)
        else:
            # extraction mode: per-pixel date
            self.datetime = None

        # Compute odr for each channel
        if odr is None:
            # Load srf for each channel
            if srf is None:
                srf = get_SRF(ds.sensor, ds.platform)
            else:
                assert isinstance(srf,xr.Dataset)
            odr = get_ODR_for_sensor(srf)
        else:
            assert isinstance(odr,xr.Dataset)
        self.odr = odr

        init_geometry(self.ds)

        self.ds.reset_coords().chunk(bands=-1)
    
    def run(self, varname='Rtoa'):
        wdspd = 5. # FIXME: use ancillary data
        attrs = {'rod_source': 'Bodhaine99'}

        coords = {
            'raa': self.ds.raa,
            'mu_s': self.ds.mus,
            'mu_v': self.ds.muv,
            'wdspd': wdspd,
            'odr': self.odr,
        }
        
        # switch to float64 because of issue with interp ;
        # the dtype is always float64 after compute, even though
        # it is float32 before compute
        # => use map_blocks instead ?
        self.ds['rho_mol_gli'] = self.lut.reflectance_toa.interp(
            coords).reset_coords(drop=True).astype('float64')
        self.ds['rho_mol_gli'].attrs.update({
            'desc': 'Rayleigh + sun glint reflectance',
            **attrs})

        self.ds['rho_mol'] = (self.lut.reflectance_toa - self.lut.reflectance_glitter).interp(
            coords).reset_coords(drop=True).astype('float64')
        self.ds['rho_mol'].attrs.update({
            'desc': 'Rayleigh reflectance (no sun glint)',
            **attrs})

        self.ds['rho_rc'] = self.ds[varname] - self.ds['rho_mol_gli']
        self.ds['rho_rc'].attrs.update({'desc': 'Rayleigh corrected reflectance',
                                **attrs})

        # Total atmospheric diffuse transmittance
        self.ds['t_d'] = (self.lut.trans_flux_down.interp(
            mu_s=self.ds.mus, odr=self.odr, wdspd=wdspd
        ) * self.lut.trans_flux_up.interp(
            mu_v=self.ds.muv, odr=self.odr)).reset_coords(drop=True).astype('float64')
        self.ds['t_d'].attrs.update(
            {'desc': 'Total atmospheric diffuse transmittance',
            **attrs})
        return 
    
    def apply(self):
        return self.run()

    

def get_ancillary(l1: xr.Dataset, list_correction:list, provider):
    list_requires = [correc.requires_anc for correc in list_correction]
    set_requires = set.union(*map,(set,list_requires))
    if len(list_requires) == 0:
        return l1
    
    date = dt.strptime(l1.datetime[:10],'%Y-%m-%d')
    for req in set_requires:
        provider.get('GACF',req, date)
    return l1


def get_ODR_for_sensor(srf:xr.Dataset):
    out_ODR = {}
    for band in srf:
        srf_band = srf[band].rename({f'wav_{band}':'wav'})
        wav = srf_band.coords['wav'].values
        od = ODR(wav*1e-3, np.array(400), 45., 0., 1013.25)
        od_xr = xr.DataArray(od.squeeze(), 
                        coords={'wav':wav})
        integrate = np.trapz(srf_band*od_xr, x=wav)
        normalize = np.trapz(srf_band,x=wav)
        out_ODR[band] = integrate / normalize
    return xr.DataArray(list(out_ODR.values()), coords={'band':list(srf.keys())})