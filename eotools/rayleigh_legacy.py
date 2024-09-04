import numpy as np
import xarray as xr
from eoread.download_legacy import download_url
from core import config
from luts import Idx, read_mlut_hdf

from eotools.bodhaine import rod

"""
Legacy Rayleigh correction

Relies on HDF4 LUT and luts objects
"""


class Rayleigh_correction:
    def __init__(self, ds, **kwargs):
        self.ds = ds
        
        lut_file = download_url(
            "https://docs.hygeos.com/s/M7iK4eX4CbpYKj8/download/LUT.hdf",
            config.get('dir_static')/'rayleigh'
        )
        self.mlut = read_mlut_hdf(str(lut_file))
        self.bitmask_invalid = -1  # FIXME:


    def run(self, wav, mus, muv, raa, altitude, surf_press, wind_speed, flags):
        '''
        Rayleigh correction
        + transmission interpolation
        '''
        nbands = wav.shape[-1]
        dim3 = mus.shape + (nbands,)
        Rmol = np.zeros(dim3, dtype='float32') + np.NaN
        Rmolgli = np.zeros(dim3, dtype='float32') + np.NaN
        Tmol = np.zeros(dim3, dtype='float32') + np.NaN

        ok = (((flags & self.bitmask_invalid) == 0)
              & (mus > 0.17)   # avoid cases where sun is too low
              & (muv > 0.17)   # ...or sensor
              )

        wind = wind_speed[ok]
        wmax = np.amax(self.mlut.axis('dim_wind'))
        wind[wind > wmax] = wmax  # clip to max wind

        mus_ = mus[ok]
        muv_ = muv[ok]
        raa_ = raa[ok]

        for i in range(nbands):

            # calculate Rayleigh optical thickness on the fly
            tau_ray = rod((wav[i] if wav.ndim == 1 else wav[ok, i])/1000.,
                          400.,
                          45.,
                          altitude[ok],
                          surf_press[ok])

            Rmolgli[ok, i] = self.mlut['Rmolgli'][
                    Idx(muv_),
                    Idx(raa_),
                    Idx(mus_),
                    Idx(tau_ray),
                    Idx(wind)]
            Rmol[ok, i] = self.mlut['Rmol'][
                    Idx(muv_),
                    Idx(raa_),
                    Idx(mus_),
                    Idx(tau_ray)]

            Tmol[ok, i]  = self.mlut['Tmolgli'][
                    Idx(mus_),
                    Idx(tau_ray),
                    Idx(wind)]
            Tmol[ok, i] *= self.mlut['Tmolgli'][
                    Idx(muv_),
                    Idx(tau_ray),
                    Idx(wind)]

        return Rmol.astype('float32'), Rmolgli.astype('float32'), Tmol.astype('float32')

    def apply(self):
        ds = self.ds
        # wav, _ = xr.broadcast(ds.wav, ds.Rtoa)    # TODO: necessary with apply_ufunc ?
        # if ds.Rtoa.chunks:
        #     wav = wav.chunk(ds.Rtoa.chunks)      # TODO: necessary ?

        Rmol, Rmolgli, Tmol = xr.apply_ufunc(
            self.run,
            ds.wav, ds.mus, ds.muv, ds.raa,
            ds.altitude, ds.sea_level_pressure, ds.horizontal_wind, ds.flags,
            dask='parallelized',
            input_core_dims=[
                ['bands'], [], [], [],
                [], [], [], [],
                ],
            output_core_dims=[['bands'], ['bands'], ['bands']],
            output_dtypes=['float32', 'float32', 'float32'],
        )

        ds['rho_r'] = Rmol
        ds['rho_r_gli'] = Rmolgli
        ds['Tmol'] = Tmol
        ds['Rprime'] = ds.rho_gc - ds.rho_r_gli
        ds['Rprime_noglint'] = ds.rho_gc - ds.rho_r

