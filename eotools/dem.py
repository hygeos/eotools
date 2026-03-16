import xarray as xr

from core.geo.naming import names
from core.download import download_url
from core.process.blockwise import BlockProcessor
from core.interpolate import Interpolator, Linear, Nearest
from core.tools import Var, xrcrop
from core import env, mdir

from typing import Any, Literal
from xarray import Dataset
from pathlib import Path


class InitAltitude(BlockProcessor):
    """
    Altitude initialization

    Currently sets altitude to zero.
    """
    def input_vars(self) -> list[Var]:
        return [names.lat, names.lon]

    def created_vars(self) -> list[Var]:
        return [Var("altitude", attrs={"units": "m"})]

    def process_block(self, block: Dataset) -> None:
        block["altitude"] = xr.zeros_like(block[str(names.lat)])
        block["altitude"].attrs.update(units='m')
        

class GTOPO30(BlockProcessor):
    """
    GTOPO30 digital elevation model

    30 arc-second (~1km) - Between 56S and 60N

    Args:
    -----
    directory: directory for tile storage
    missing: float to provide in case of missing value
    """
    
    def __init__(self, 
            directory: str|Path|None = None, 
            missing: Any = 0,
            method: Literal['nearest','linear'] = 'linear'
        ):
        self.missing = missing
        
        if method == 'linear': self.method = Linear
        elif method == 'nearest': self.method = Nearest
        else: raise ValueError
        
        if directory is None:
            directory = mdir(env.getdir("DIR_STATIC") / "GTOPO30")
        else:
            directory = Path(directory)

        url = 'http://download.hygeos.com/eoread/GTOPO30_DZ_MLUT.nc'  
        gtopo_file = download_url(url, directory, if_exists='skip')
        self.dem = xr.open_dataset(gtopo_file, engine='h5netcdf')
    
    def input_vars(self) -> list[Var]:
        return [names.lat, names.lon]

    def created_vars(self) -> list[Var]:
        return [Var("altitude", attrs={"units": "m"})]
    
    def process_block(self, block):
        dem = xrcrop(self.dem, lat=block[str(names.lat)], lon=block[str(names.lon)])
        params = dict(lat=self.method(str(names.lat)), lon=self.method(str(names.lon)))
        Interpolator(dem, **params).process_block(block)
        block["altitude"] = xr.where(dem.elev>0, dem.elev, self.missing)