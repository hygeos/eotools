import numpy as np
import xarray as xr
from eoread.download_legacy import download_url
from eoread.utils.config import load_config
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
            load_config()['dir_static']/'rayleigh'
        )
        self.mlut = read_mlut_hdf(str(lut_file))


    def run(self, wav, mus, muv, raa, altitude, surf_press, wind_speed, flags):
        '''
        Rayleigh correction
        + transmission interpolation
        '''
        nbands = wav.shape[-1]
        Rmol = np.zeros_like(wav) + np.NaN
        Rmolgli = np.zeros_like(wav) + np.NaN
        Tmol = np.zeros_like(wav) + np.NaN

        ok = (((flags & self.ds.BITMASK_INVALID) == 0)
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
            tau_ray = rod(wav[ok, i]/1000., 400., 45.,
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

        ds['Rmol'] = Rmol
        ds['Rmolgli'] = Rmolgli
        ds['Tmol'] = Tmol
        ds['Rprime'] = ds.Rtoa_gc - ds.Rmolgli
        ds['Rprime_noglint'] = ds.Rtoa_gc - ds.Rmol
