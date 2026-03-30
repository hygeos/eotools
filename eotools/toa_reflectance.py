from datetime import datetime

import numpy as np
import xarray as xr
from core.process.blockwise import BlockProcessor, Var
from core.geo.naming import names
from eotools.solar_irradiance import solar_irradiance_lisird
from eotools.srf import integrate_srf
from core.tools import datetime as datetime_parse
from eotools.units import convert


def sun_earth_distance(t: datetime) -> float:
    """Return the sun-earth distance in AU for a given datetime.

    d is computed using a two-term Keplerian expansion of Earth's orbit.
    Use it in the TOA reflectance formula as:
        rho = pi * L * sun_earth_distance(t)**2 / (F0 * cos(sza))
    where F0 is the solar irradiance tabulated at 1 AU.

    Args:
        t: acquisition datetime (timezone-aware or naive, UTC assumed).

    Returns:
        Sun-earth distance in AU.
    """
    A = 1.00014
    B = 0.01671
    C = 0.9856002831  # deg/day  (= 360 / 365.25)
    D = 3.4532858  # places perihelion at ~Jan 3-4
    F = 0.00014  # second-harmonic amplitude (~e²/2)

    jday = t.timetuple().tm_yday
    M = 2 * np.pi * (C * jday - D) / 360.0
    return A - B * np.cos(M) - F * np.cos(2 * M)


class Init_rho_toa(BlockProcessor):
    """BlockProcessor that computes TOA reflectance from TOA radiance.

    Implements the standard formula:
        rho_toa = pi * L_toa * d^2 / (mu_s * F0)

    where d is the sun-earth distance in AU, mu_s = cos(SZA), and F0 is the
    extraterrestrial solar irradiance at 1 AU.

    The processor is a no-op if ``rtoa`` is already present in the dataset.
    F0 can come from two sources (in order of priority):
    1. As a per-pixel band variable in the dataset (``names.F0``).
    2. Integrated from ``srf`` using a solar spectrum: either the
       ``solar_irradiance`` argument or, by default, the LISIRD p1nm spectrum.

    Args:
        ds: Input dataset used to determine which variables are already present
            and to parse the acquisition datetime.
        srf: Per-band spectral response functions used to integrate F0 when it
            is not available in ``ds``.  Required when ``names.F0`` is absent
            from ``ds``.
        solar_irradiance: Solar spectrum DataArray (wavelength axis named
            ``"wav"``) to integrate over ``srf``.  Defaults to the LISIRD
            p1nm spectrum when ``None``.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        srf: xr.Dataset | None = None,
        solar_irradiance: xr.DataArray | None = None,
    ):
        self.enable = names.rtoa not in ds
        if self.enable:
            self.has_F0 = names.F0 in ds

            dt = datetime_parse(ds)
            self.d = sun_earth_distance(dt)

            if not self.has_F0:
                # Solar irradiance is not provided: calculate it from srf
                assert srf is not None
                if solar_irradiance is None:
                    F0_data = solar_irradiance_lisird("p1nm").SSI.rename(
                        wavelength="wav"
                    )
                else:
                    F0_data = solar_irradiance
                F0_integrated = integrate_srf(srf, F0_data, resample="srf")
                self.F0 = xr.concat(
                    [F0_integrated[x] for x in F0_integrated],
                    dim="bands",
                ).assign_coords(bands=list(srf))

    def input_vars(self) -> list[Var]:
        if self.enable:
            if self.has_F0:
                return [Var(names.ltoa), Var(names.sza), Var(names.F0)]
            else:
                return [Var(names.ltoa), Var(names.sza)]
        else:
            return []

    def created_vars(self) -> list[Var]:
        if self.enable:
            return [Var(names.rtoa)]
        else:
            return []

    def check(self, ds: xr.Dataset) -> None:
        assert (ds.bands.values == self.F0.bands.values).all()

    def auto_template(self) -> bool:
        return True

    def process_block(self, block: xr.Dataset) -> None:
        sza = block[names.sza]
        mus = np.cos(np.radians(sza))
        Ltoa = block[names.ltoa]

        if self.has_F0:
            F0 = block[names.F0]
        else:
            F0 = self.F0

        F0_converted = convert(F0, Ltoa.units)
        d = self.d

        block[str(names.rtoa)] = np.pi * Ltoa * d**2 / (mus * F0_converted)
