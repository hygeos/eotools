
import xarray as xr
import numpy as np

from core.process.blockwise import BlockProcessor
from core.tools import Var
from core.geo import n


def scattering_angle(mu_s, mu_v, phi):
    """
    Scattering angle in degrees

    mu_s: cos of the sun zenith angle (degrees)
    mu_v: cos of the view zenith angle (degrees)
    phi: relative azimuth angle in degrees
    """
    sa = -mu_s*mu_v - np.sqrt((1.-mu_s*mu_s)*(1.-mu_v*mu_v)) * np.cos(np.radians(phi))
    return np.arccos(sa)*180./np.pi


class InitGeometry(BlockProcessor):
    """
    Initialize geometrical variables:
        - cosines of the view and zenith angles (muv, mus)
        - relative azimuth angle (raa)
        - air mass (1/mus + 1/muv)
        - scattering angle
    """
    def __init__(
        self,
        ds: xr.Dataset,
        calc_scat_angle: bool = False,
        calc_air_mass: bool = False,
    ):
        self.calc_scat_angle = calc_scat_angle
        self.calc_air_mass = calc_air_mass
        self.calc_raa = n.raa not in ds

    def input_vars(self) -> list[Var]:
        ivars = [Var("sza"), Var("vza")]
        if self.calc_raa:
            ivars.append( Var("vaa"))
            ivars.append( Var("saa"))
        return ivars

    def created_vars(self) -> list[Var]:
        cvars = [
            Var(str(n.mus)),
            Var(str(n.muv)),
            Var(str(n.raa)),
        ]
        if self.calc_scat_angle:
            cvars.append(Var("scat_angle"))
        if self.calc_air_mass:
            cvars.append(Var("air_mass"))
        return cvars

    def process_block(self, block: xr.Dataset):
        # calculate mus and muv
        block[str(n.mus)] = np.cos(np.radians(block.sza))
        block[str(n.mus)].attrs['description'] = n.mus.desc
        block[str(n.muv)] = np.cos(np.radians(block.vza))
        block[str(n.muv)].attrs['description'] = n.muv.desc

        # relative azimuth angle
        raa = block[str(n.saa)] - block[str(n.vaa)]
        raa = raa % 360
        block[str(n.raa)] = raa.where(raa < 180, 360 - raa)
        block[str(n.raa)].attrs['description'] = n.raa.desc
        block[str(n.raa)].attrs['units'] = n.raa.unit

        # scattering angle
        if self.calc_scat_angle:
            block['scat_angle'] = scattering_angle(block[str(n.mus)], block[str(n.muv)], block[str(n.raa)])
            block['scat_angle'].attrs['description'] = 'scattering angle'
            block['scat_angle'].attrs['units'] = 'degrees'
        
        # air mass
        if self.calc_air_mass:
            block['air_mass'] = 1 / block[str(n.mus)] + 1 / block[str(n.muv)]
            block['air_mass'].attrs['description'] = 'air mass'
