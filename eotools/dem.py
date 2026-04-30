import xarray as xr

from core.geo.naming import names
from core.network.download import download_url
from core.process.blockwise import BlockProcessor
from core.interpolate import Interpolator, Linear, Nearest
from core.tools import Var, xrcrop
from core import env, mdir

from typing import Any, Literal, Tuple
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

    def auto_template(self) -> bool:
        return True

    def process_block(self, block: Dataset) -> None:
        block["altitude"] = xr.zeros_like(block[str(names.lat)])
        block["altitude"].attrs.update(units="m")


class GTOPO30(BlockProcessor):
    """
    GTOPO30 digital elevation model

    30 arc-second (~1km) - Between 56S and 60N

    Args:
    -----
    l1: Xarray Dataset for an eoread reader
    lat: tuple of (min, max) latitude bounds
    lon: tuple of (min, max) longitude bounds
    directory: directory for tile storage
    missing: value to use for missing elevation data (default: 0)
    method: interpolation method, 'nearest' or 'linear' (default: 'linear')
    """

    def __init__(
        self,
        l1: xr.Dataset = None,
        lat: Tuple[float] = None,
        lon: Tuple[float] = None,
        directory: str | Path | None = None,
        missing: Any = 0,
        method: Literal["nearest", "linear"] = "linear",
        verbose: bool = True,
    ):
        self.missing = missing

        if method == "linear":
            self.method = Linear
        elif method == "nearest":
            self.method = Nearest
        else:
            raise ValueError

        if directory is None:
            directory = mdir(env.getdir("DIR_STATIC") / "GTOPO30")
        else:
            directory = Path(directory)

        # Download the GTOPO file
        url = "http://download.hygeos.com/eoread/GTOPO30_DZ_MLUT.nc"
        gtopo_file = download_url(url, directory, if_exists="skip", verbose=verbose)
        dem = xr.open_dataset(gtopo_file, engine="h5netcdf")

        # Crop it to avoid loading all the raster
        if l1:
            self.dem = xrcrop(dem, lat=l1[str(names.lat)], lon=l1[str(names.lon)])

        elif lat or lon:
            assert lat and lon, "Latitude and longitude constraints should be provided"
            dims = (str(names.rows), str(names.columns))
            lon = xr.DataArray([lon] * 2, dims=dims)
            lat = xr.DataArray([lat] * 2, dims=dims).T
            self.dem = xrcrop(dem, lat=lat, lon=lon)

    def input_vars(self) -> list[Var]:
        return [names.lat, names.lon]

    def created_vars(self) -> list[Var]:
        return [Var("altitude", dims_like=str(names.lat), attrs={"units": "m"})]

    def auto_template(self) -> bool:
        return True

    def process_block(self, block):
        assert hasattr(self, "dem"), "Provide LatLon constraints in constructor"
        params = dict(lat=self.method(str(names.lat)), lon=self.method(str(names.lon)))
        Interpolator(self.dem, **params).process_block(block)
        block["altitude"] = xr.where(block.elev > 0, block.elev, self.missing)
