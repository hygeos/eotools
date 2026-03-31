from pathlib import Path
from typing import Literal
from core.tools import Var
import numpy as np
import xarray as xr
from eotools.srf import integrate_srf
from eotools.bodhaine import rod, raycrs, column_number_density
from eotools.chapman import chapman
from core.process.blockwise import BlockProcessor
from core.interpolate import Interpolator
from core.download import download_url
from core import env
from eotools.units import check_units, convert
from eotools.utils.tools import deduplicate_dims
from luts import read_mlut_hdf

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

    Computes the Rayleigh optical depth based on altitude and pressure.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing altitude and pressure information.
        Must contain either:
        - "altitude" and "surface_pressure" or "sea_level_pressure"
        - "cwav" (central wavelength) for single-wavelength calculation
    srf : xr.Dataset | None, optional
        Sensor spectral response function for multi-wavelength calculation.
        If provided, ODR is calculated by integrating the Rayleigh cross-section
        over the SRF.

    Returns
    -------
    xr.DataArray
        Rayleigh optical depth values with same dimensions as input.

    Notes
    -----
    - If srf is provided, ODR is computed by integrating the Rayleigh cross-section
      over the SRF using the Bodhaine et al. (1999) method.
    - If srf is None, ODR is computed using the central wavelength from "cwav"
    """
    altitude_m = convert(ds["altitude"], "m")
    if "surface_pressure" in ds:
        pressure_hpa = convert(ds["surface"], "hPa")
        pressure_kind = "surface"
    else:
        pressure_hpa = convert(ds["sea_level_pressure"], "hPa")
        pressure_kind = "sea-level"

    if srf is not None:
        # calculate ODR from SRF integration
        for coord in srf.coords:
            check_units(srf.coords[coord], 'nm')
        ray_crs = integrate_srf(srf, lambda lam_nm: raycrs(lam_nm/1000.))
        ray_crs = xr.concat([ray_crs[x] for x in ray_crs], dim='bands')

        cnd = column_number_density(
            z=altitude_m, P=pressure_hpa, pressure=pressure_kind
        )

        # ray_crs is wavelength dependent
        # cnd has image dimensions (altitude, pressure)
        odr = ray_crs * cnd

        return odr

    elif "cwav" in ds:
        # calculate ODR from central wavelength "wav" provided as input
        if hasattr(ds["cwav"], "units"):
            check_units(ds['cwav'], 'nm')
        odr = rod(
            ds['cwav'] / 1000.0, z=altitude_m, P=pressure_hpa, pressure=pressure_kind
        )
        ds.attrs.update({"rod_source": "Bodhaine99"})
        ds["odr"] = odr
        return odr

    else:
        raise RuntimeError("Unable to calculate ODR.")


def rayleigh_correction(ds: xr.Dataset, srf: xr.Dataset | None = None, **cfg):
    """
    Apply Rayleigh correction to `ds`

    This function applies atmospheric Rayleigh scattering correction to the
    top-of-atmosphere reflectance data. It computes the Rayleigh-corrected
    reflectance and related atmospheric transmission parameters.

    This creates the following variables in `ds`:

    +-------------+-----------------------------------------------+
    | Variable    | Description                                   |
    +=============+===============================================+
    | rho_rc      | Rayleigh-corrected signal                     |
    +-------------+-----------------------------------------------+
    | rho_mol_gli | Rayleigh + glint reflectance                  |
    +-------------+-----------------------------------------------+
    | rho_mol     | Rayleigh reflectance (no glint)               |
    +-------------+-----------------------------------------------+
    | t_d         | Total diffuse transmittance (downward+upward) |
    +-------------+-----------------------------------------------+

    Parameters
    ----------
    ds : xr.Dataset
        The Dataset used for inputs/outputs. Must contain:
        - "altitude": altitude above sea level
        - "horizontal_wind": horizontal wind speed
        - "mus": cosine of solar zenith angle
        - "muv": cosine of viewing zenith angle
        - "raa": relative azimuth angle (saa - vaa)
        - "rho_gc": ground-corrected reflectance
        - "surface_pressure" or "sea_level_pressure": atmospheric pressure
    srf : xr.Dataset | None, optional
        The sensor spectral response function (SRF).
    **cfg : dict
        Additional configuration parameters passed to `load_rayleigh_lut`.

    Returns
    -------
    xr.Dataset
        The input dataset with added variables: rho_rc, rho_mol_gli, rho_mol, t_d.
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


def read_lut_polymer_legacy() -> xr.Dataset:
    lut_file = download_url(
        "https://docs.hygeos.com/s/M7iK4eX4CbpYKj8/download/LUT.hdf",
        env.getdir('DIR_STATIC')/'rayleigh'
    )

    # TODO: use xr.open_dataset(engine='netcdf4') to avoid mlut dependency
    rayleigh_lut = read_mlut_hdf(str(lut_file)).to_xarray()

    # Remove duplicate dimensions
    rayleigh_lut_dedup = deduplicate_dims(
        rayleigh_lut[["Rmol", "Rmolgli"]], {"dim_mu": ["mu_s", "mu_v"]}
    )

    # rayleigh_lut_dedup['Tmolgli'] = rayleigh_lut['Tmolgli'].rename({'dim_mu': 'mu'})
    t_d_down = rayleigh_lut['Tmolgli'].rename(dim_mu='mu_s')
    t_d_up = rayleigh_lut['Tmolgli'].rename(dim_mu='mu_v')

    rayleigh_lut_dedup['t_d'] = t_d_down * t_d_up

    return rayleigh_lut_dedup.rename(
        dim_phi="raa",
        dim_tauray="odr",
        dim_wind="wind_speed",
        Rmol="rho_r",
        Rmolgli="rho_rg",
    )


class RayleighCorrection(BlockProcessor):
    """
    Block processor for applying Rayleigh scattering correction.

    This processor applies atmospheric Rayleigh scattering correction to
    satellite reflectance data using pre-computed lookup tables (LUTs).

    Parameters
    ----------
    srf : xr.Dataset | None, optional
        Sensor spectral response function for multi-wavelength correction.
    version : Literal["polymer_legacy"], default="polymer_legacy"
        Version of the Rayleigh LUT to use. Currently only "polymer_legacy"
        is supported.
    pressure_kind : Literal["surface_pressure", "sea_level_pressure"], default="sea_level_pressure"
        Type of pressure data to use. "sea_level_pressure" uses sea-level
        pressure, "surface_pressure" uses surface pressure.
    **cfg : dict
        Additional configuration parameters passed to `load_rayleigh_lut`.

    Attributes
    ----------
    rayleigh_lut : xr.Dataset
        Pre-computed Rayleigh scattering lookup table.
    interpolator : Interpolator
        Interpolator object for performing multi-dimensional interpolation.
    pressure_kind : str
        Type of pressure data used.
    """

    def __init__(
        self,
        srf: xr.Dataset | None = None,
        version: Literal["polymer_legacy"] = "polymer_legacy",
        pressure_kind: Literal[
            "surface_pressure", "sea_level_pressure"
        ] = "sea_level_pressure",
        sun_glint_corr: bool = True,
        **cfg,
    ):
        """
        Initialize the Rayleigh correction processor.

        Parameters
        ----------
        srf : xr.Dataset | None, optional
            Sensor spectral response function.
        version : Literal["polymer_legacy"], default="polymer_legacy"
            LUT version to use.
        pressure_kind : Literal["surface_pressure", "sea_level_pressure"], default="sea_level_pressure"
            Pressure data type used at input.
        sun_glint_corr : bool, default=True
            Whether to include sun glint correction in the Rayleigh correction.
            If True, corrects for both Rayleigh scattering and sun glint.
            If False, corrects only for Rayleigh scattering.
        **cfg : dict
            Additional configuration.
        """
        # todo: bitmask invalid
        if version == 'polymer_legacy':
            self.rayleigh_lut = read_lut_polymer_legacy()
        self.srf = srf
        self.sun_glint_corr = sun_glint_corr
        self.interpolator = Interpolator(
            self.rayleigh_lut,
            odr=Linear("odr"),
            mu_s=Linear("mus"),
            mu_v=Linear("muv"),
            raa=Linear("raa"),
            wind_speed=Linear("horizontal_wind"),
        )
        self.pressure_kind = pressure_kind
    
    def input_vars(self) -> list[Var]:
        ivars = [
            Var("altitude"),
            Var("horizontal_wind"),
            Var(self.pressure_kind),
            Var("mus"),
            Var("muv"),
            Var("raa"),
            Var("rho_gc"),
        ]
        if self.srf is None:
            ivars.append(Var("cwav"))
        return ivars
    
    def created_vars(self) -> list[Var]:
        return [
            Var(
                "rho_r",
                dtype="float64",
                dims_like="rho_gc",
                attrs={"desc": "Rayleigh reflectance"},
            ),
            Var(
                "rho_rg",
                dtype="float64",
                dims_like="rho_gc",
                attrs={"desc": "Rayleigh + sun glint reflectance"},
            ),
            Var(
                "t_d",
                dtype="float64",
                dims_like="rho_gc",
                attrs={
                    "desc": "Total transmittance (upward, downward, direct and diffuse)"
                },
            ),
            Var(
                "rho_rc",
                dtype="float64",
                dims_like="rho_gc",
                attrs={"desc": "Rayleigh corrected reflectance"},
            ),
        ]
    
    def check(self, ds: xr.Dataset) -> None:
        """
        Check that srf bands are identical to ds.bands
        """
        if self.srf is not None:
            ds_bands = [str(x) for x in ds.bands.values]
            srf_bands = [str(x) for x in self.srf]
            assert ds_bands == srf_bands
        check_units(ds['horizontal_wind'], 'm/s')
    
    def process_block(self, block: xr.Dataset):
        block['odr'] = calc_odr(block, self.srf).transpose(*block.rho_gc.dims)

        self.interpolator.process_block(block)

        if self.sun_glint_corr:
            # Rayleigh + glint correction
            block['rho_rc'] = block['rho_gc'] - block['rho_rg']
        else:
            # Rayleigh only correction
            block['rho_rc'] = block['rho_gc'] - block['rho_r']
