'''
 Copyright (c) 2024 EUMETSAT
 License: MIT
'''

from pathlib import Path
from typing import Literal
import numpy as np
import xarray as xr
from eotools.srf import integrate_srf
from eotools.bodhaine import rod
from eotools.chapman import chapman

from core.interpolate import interp, Linear



def load_rayleigh_lut(lut_rayleigh_pp: Path,
                      lut_rayleigh_sp: Path,
                      rayleigh: Literal["PP", "SP"],
                      **kwargs) -> xr.Dataset:
    """Load the Rayleigh LUT (PP or SP)
    
    Relative azimuth angle convention: standard (raa = saa - vaa)

    Returns a dataset with:
        - reflectance_glitter (reflectance with a rough sea surface, no atmosphere)
        - reflectance_toa (including glitter)
        - T_tot_down: total transmission (downward)
        - T_tot_up: total transmission (upward)
    """
    lut = xr.Dataset()

    PP = xr.open_dataset(lut_rayleigh_pp,
                            chunks=-1)

    # set standard convention, raa = saa - vaa
    PP = PP.assign_coords(raa=180 - PP.raa)

    if rayleigh == 'PP':

        lut['reflectance_glitter'] = PP['reflectance_glitter']
        lut['reflectance_toa'] = PP['reflectance_toa']
        lut['T_tot_down'] = PP['trans_flux_down']
        lut['T_tot_up'] = PP['trans_flux_up']

    elif rayleigh == 'SP':

        SP = xr.open_dataset(lut_rayleigh_sp,
                             chunks=-1)

        SP = SP.rename({'mus_s': 'mu_s', 'mus_v': 'mu_v'})

        # set standard convention, raa = saa - vaa
        SP = SP.assign_coords(raa=180 - SP.raa)

        lut['reflectance_glitter'] = SP['rho_total'] - SP['rho_path']
        lut['reflectance_toa'] = SP['rho_total']

        # Apply spherical correction to the direct term of the PP transmission
        lut["T_tot_down"] = (
            PP["trans_flux_down"]
            - np.exp(-PP.odr * (1 / PP.mu_s))
            + np.exp(-PP.odr * chapman(PP.mu_s))
        ).interp(odr=SP.odr, mu_s=SP.mu_s)
        lut["T_tot_up"] = (
            PP["trans_flux_up"]
            - np.exp(-PP.odr * (1 / PP.mu_v))
            + np.exp(-PP.odr * chapman(PP.mu_v))
        ).interp(odr=SP.odr, mu_v=SP.mu_v)
    
    else:
        raise ValueError(rayleigh)

    return lut.compute(scheduler="sync")


def calc_odr(ds: xr.Dataset, srf: xr.Dataset | None = None) -> xr.DataArray:
    """
    Calculates ODR (Optical Depth Rayleigh)
    """
    def func_rod(wav_nm):
        return rod(wav_nm / 1000)

    if "odr" in ds:
        # TODO: make it pixel by pixel, account for surface pressure
        assert ds.odr.ndim == 1
        ds.attrs.update({"rod_source": str(ds.odr.values)})
        return ds["odr"]

    elif srf is not None:
        # calculate ODR from SRF
        odr_dict = integrate_srf(srf, func_rod)
        odr = xr.DataArray([odr_dict[b] for b in ds.bands.values], dims="bands")
        ds.attrs.update({"rod_source": "Bodhaine99"})
        ds["odr"] = odr
        return odr

    elif "wav" in ds:
        # calculate ODR from central wavelength "wav"
        assert ds.wav.units == "nm"
        odr = func_rod(ds.wav)
        ds.attrs.update({"rod_source": "Bodhaine99"})
        ds["odr"] = odr
        return odr

    else:
        raise RuntimeError("Unable to calculate ODR.")


def rayleigh_correction(ds: xr.Dataset, srf: xr.Dataset | None = None, **cfg):
    """Apply Rayleigh correction to `ds`
    
    This creates the following variables in `ds`:

    +-------------+-----------------------------------------------+
    | Variable    | Description                                   |
    +=============+===============================================+
    | rho_rc      | Rayleigh-corrected signal                     |
    +-------------+-----------------------------------------------+
    | rho_mol_gli | Rayleigh + glint reflectance                  |
    +-------------+-----------------------------------------------+
    | rho_mod     | Rayleigh reflectance (no glint)               |
    +-------------+-----------------------------------------------+
    | t_d         | Total diffuse transmittance (downward+upward) |
    +-------------+-----------------------------------------------+

    Parameters
    ----------
    ds : xr.Dataset
        The Dataset used for inputs/outputs.
    srf : xr.Dataset
        The sensor spectral response function (SRF).
    """

    lut = load_rayleigh_lut(**cfg)

    odr = calc_odr(ds, srf)

    wdspd = ds["horizontal_wind"]

    coords = {
        "raa": Linear(ds.raa),
        "mu_s": Linear(ds.mus, bounds='nan'),
        "mu_v": Linear(ds.muv, bounds='nan'),
        "wdspd": Linear(wdspd, bounds='clip'),
        "odr": Linear(odr),
    }

    rho_toa = lut.reflectance_toa
    rho_gli = lut.reflectance_glitter
    ds["rho_mol_gli"] = interp(
        rho_toa,
        **coords,
    )
    ds["rho_mol_gli"].attrs.update(
        {"desc": "Rayleigh + sun glint reflectance"}
    )

    ds["rho_mol"] = interp(
        rho_toa - rho_gli,
        **coords,
    )
    ds["rho_mol"].attrs.update({"desc": "Rayleigh reflectance (no sun glint)"})

    ds["rho_rc"] = ds["rho_gc"] - ds["rho_mol_gli"]
    ds["rho_rc"].attrs.update({"desc": "Rayleigh corrected reflectance"})

    # Total atmospheric diffuse transmittance
    ds["t_d"] = interp(
        lut.T_tot_down,
        mu_s=Linear(ds.mus, bounds="nan"),
        odr=Linear(odr),
        wdspd=Linear(wdspd, bounds="clip"),
    ) * interp(
        lut.T_tot_up,
        mu_v=Linear(ds.mus, bounds="nan"),
        odr=Linear(odr),
    )
    ds["t_d"].attrs.update({"desc": "Total atmospheric transmittance"})
