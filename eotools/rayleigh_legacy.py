import numpy as np
import xarray as xr
from core.download import download_url
from core import env
from core.tools import MapBlocksOutput, Var
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
            env.getdir('DIR_STATIC')/'rayleigh'
        )
        self.mlut = read_mlut_hdf(str(lut_file))
        self.bitmask_invalid = -1  # FIXME:
        self.model = MapBlocksOutput([
            Var('Rmol', 'float32', ('y', 'x', 'bands')),
            Var('Rmolgli', 'float32', ('y', 'x', 'bands')),
            Var('Tmol', 'float32', ('y', 'x', 'bands')),
        ])

    def run(self, wav, mus, muv, raa, altitude, surf_press, wind_speed, flags):
        '''
        Rayleigh correction
        + transmission interpolation
        '''
        nbands = wav.shape[-1]
        dim3 = mus.shape + (nbands,)
        Rmol = np.zeros(dim3, dtype='float32') + np.nan
        Rmolgli = np.zeros(dim3, dtype='float32') + np.nan
        Tmol = np.zeros(dim3, dtype='float32') + np.nan

        if self.bitmask_invalid >= 0:
            ok = (((flags & self.bitmask_invalid) == 0)
                & (mus > 0.17)   # avoid cases where sun is too low
                & (muv > 0.17)   # ...or sensor
                )
        else:
            ok = ((mus > 0.17)   # avoid cases where sun is too low
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

    def apply_block(self, ds: xr.Dataset):
        Rmol, Rmolgli, Tmol = self.run(
            ds.wav.transpose(..., "bands").data,
            ds.mus.data,
            ds.muv.data,
            ds.raa.data,
            ds.altitude.data,
            ds.surf_press.data,
            ds.wind_speed.data,
            ds.flags.data,
        )
        out = xr.Dataset()
        out['Rmol'] = (('y', 'x', 'bands'), Rmol)
        out['Rmolgli'] = (('y', 'x', 'bands'), Rmolgli)
        out['Tmol'] = (('y', 'x', 'bands'), Tmol)
        out = out.assign_coords(bands=ds.bands.values)

        return self.model.conform(out)

    def apply(self, method='apply_ufunc'):
        ds = self.ds
        # wav, _ = xr.broadcast(ds.wav, ds.Rtoa)    # TODO: necessary with apply_ufunc ?
        # if ds.Rtoa.chunks:
        #     wav = wav.chunk(ds.Rtoa.chunks)      # TODO: necessary ?

        if method == 'apply_ufunc':
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
        else:
            ds_out = xr.map_blocks(
                self.apply_block,
                xr.Dataset({
                    "wav": ds.wav,
                    "mus": ds.mus,
                    "muv": ds.muv,
                    "raa": ds.raa,
                    "altitude": ds.altitude,
                    "surf_press": ds.sea_level_pressure,
                    "wind_speed": ds.horizontal_wind,
                    "flags": ds.flags,
                }),
                template=self.model.template(ds),
            )
            Rmol = ds_out['Rmol']
            Rmolgli = ds_out['Rmolgli']
            Tmol = ds_out['Tmol']

        ds['rho_r'] = Rmol
        ds['rho_r_gli'] = Rmolgli
        ds['Tmol'] = Tmol
        ds['Rprime'] = ds.rho_gc - ds.rho_r_gli
        ds['Rprime_noglint'] = ds.rho_gc - ds.rho_r
