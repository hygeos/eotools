import xarray as xr
import numpy as np

from eoread.eo import datetime, init_geometry
from eoread.download import FTP, get_auth_ftp, ftp_download
from pathlib import Path
from eotools.srf import get_SRF
from eotools.odr import get_ODR_for_sensor


class Rayleigh_correction:
    '''
    Rayleigh correction module

    Ex: Rayleigh_correction(l1).apply()
    '''

    requires_anc = []

    def __init__(self, ds, odr=None, srf=None):
        self.ds = ds
        self.bands = np.array(ds.bands.data)

        # Download and load Lookup Table
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
            if srf is None:
                srf = get_SRF(l1_ds=ds)
            else:
                assert isinstance(srf,xr.Dataset)
            odr = get_ODR_for_sensor(srf)
        else:
            assert isinstance(odr,xr.Dataset)
        self.odr = odr
        bands = [self.bands[(np.abs(self.bands - b)).argmin()] for b in odr.bands.values]
        self.odr = odr.assign_coords({'bands':bands})

        # Initialize angles geometry
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
        return self.ds['rho_rc']
    
    def apply(self):
        return self.run()