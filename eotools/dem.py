import xarray as xr
import numpy as np

from core.geo.naming import names
from core.network.download import download_url
from core.process.blockwise import BlockProcessor
from core.interpolate import Interpolator, Linear, Nearest
from core.tools import Var, xrcrop, drop_unused_dims
from core import env, mdir

from dask.array import concatenate, from_array, zeros
from typing import Any, Literal, Tuple, List
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
            dem = xrcrop(dem, lat=l1[str(names.lat)], lon=l1[str(names.lon)])

        elif lat or lon:
            assert lat and lon, "Latitude and longitude constraints should be provided"
            dims = (str(names.rows), str(names.columns))
            lon = xr.DataArray([lon] * 2, dims=dims)
            lat = xr.DataArray([lat] * 2, dims=dims).T
            dem = xrcrop(dem, lat=lat, lon=lon)

        else:
            raise ValueError("Provide latlon constraints to reduce memory usage")

        self.dem = dem.compute()

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


def _copdem_url_prefix(resolution: Literal[30, 90]) -> str:
    """Return the address of Copernicus DEM API server based on resolution"""
    return f"https://copernicus-dem-{resolution}m.s3.eu-central-1.amazonaws.com"


class CopernicusDEM(BlockProcessor):
    """
    Copernicus digital elevation model, 90m or 30m, with functions
    to get the altitude for a lat-lon grid and to download on demand DEM tiles
    via internet.

    Args:
    -----
    lat: tuple of (min, max) latitude bounds
    lon: tuple of (min, max) longitude bounds
    directory: directory for tile storage
    agg: Factor of aggregation

    Returns:
    -------

    A xarray.DataArray of the water occurrence between 0 and 100
    """

    def __init__(
        self,
        l1: xr.Dataset = None,
        lat: Tuple[float] = None,
        lon: Tuple[float] = None,
        directory: str | Path | None = None,
        resolution: Literal[30, 90] = 90,
        missing: Any = 0,
        method: Literal["nearest", "linear"] = "linear",
        verbose: bool = True,
    ):

        if directory is None:
            directory = mdir(env.getdir("DIR_ANCILLARY") / "COPDEM")
        else:
            directory = Path(directory)

        # Determine method to use for interpolation
        self.missing = missing
        if method == "linear":
            self.method = Linear
        elif method == "nearest":
            self.method = Nearest
        else:
            raise ValueError

        # create GSW mosaic
        mosaic = generate_full_gsw_mosaic(directory, resolution, verbose)

        # Update raster with accurated coordinates
        dims = ("lat", "lon")
        coords = dict(
            lon=np.linspace(-180, 180, mosaic.shape[1]),
            lat=np.linspace(-90, 90, mosaic.shape[0]),
        )
        raster = xr.DataArray(mosaic, name="altitude", dims=dims, coords=coords)

        # Crop it to avoid loading all the raster
        if l1:
            dem = xrcrop(raster, lat=l1[str(names.lat)], lon=l1[str(names.lon)])

        elif lat or lon:
            assert lat and lon, "Latitude and longitude constraints should be provided"
            dims = (names.rows, names.columns)
            lon = xr.DataArray([lon] * 2, dims=(names.rows, names.columns))
            lat = xr.DataArray([lat] * 2, dims=(names.columns, names.rows))
            dem = xrcrop(raster, lat=lat, lon=lon)

        else:
            raise ValueError("Provide latlon constraints to reduce memory usage")

        # Download needed GSW tiles
        self.dem = dem.compute(scheduler="synchronous")

    def input_vars(self) -> list[Var]:
        return [names.lat, names.lon]

    def created_vars(self) -> list[Var]:
        return [Var("altitude", dims_like=str(names.lat), attrs={"units": "m"})]

    def process_block(self, block: xr.Dataset) -> None:
        assert hasattr(self, "dem"), "Provide LatLon constraints in constructor"
        params = dict(lat=self.method(str(names.lat)), lon=self.method(str(names.lon)))
        Interpolator(self.dem.to_dataset(), **params).process_block(block)
        block["altitude"] = xr.where(
            ~block.altitude.isnull(), block.altitude, self.missing
        )


class _CopDEM_tile:
    pattern = "Copernicus_DSM_COG_{}_{}_00_{}_00_DEM"

    def __init__(self, lat: str, lon: str, resolution: int, directory: Path):

        N = int(1200 * 90 / resolution)
        self.empty = False
        self.shape = (N, N)
        self.dtype = "float32"
        self.tile_name = self.pattern.format(resolution // 3, lat, lon)
        self.resolution = resolution
        self.directory = directory

        if not directory.exists():
            raise IOError(
                f"Directory {directory} does not exist. "
                "It will be used to store GSW tiles. "
                "Please create it or link it first."
            )

    def set_as_empty(self):
        self.empty = True

    def __getitem__(self, key):
        return self.fetch_gsw_tile()[key].compute(scheduler="sync").values

    def fetch_gsw_tile(self) -> xr.DataArray:
        """
        Read remote file and returns its content as a numpy array
        """

        if self.empty:
            return xr.DataArray(zeros(self.shape, dtype=self.dtype) + np.nan)

        # Download tiles
        prefix = _copdem_url_prefix(self.resolution)
        url = "/".join([prefix, self.tile_name, self.tile_name + ".tif"])

        # Download and read geotiff data
        p = download_url(url, self.directory, if_exists="skip", verbose=False)
        data = xr.open_dataarray(p, engine="rasterio").squeeze()
        data = data.isel(y=slice(None, None, -1))  # Flip values along y-axis
        data = data.rename(x=str(names.columns), y=str(names.rows))
        data = drop_unused_dims(data)

        # Add relevant information
        data.attrs["source_file"] = url
        data.name = "altitude"

        return data.astype(self.dtype)


def _list_available_tiles(
        directory: Path, resolution: int, verbose: bool = True
    ) -> np.ndarray:
    """Query the list of available tiles"""
    list_filename = "tileList.txt"
    local_list_path = directory / f"tileList_{resolution}.txt"
    url = "{}/{}".format(_copdem_url_prefix(resolution), list_filename)
    if not local_list_path.exists():
        p = download_url(url, directory, if_exists="overwrite", verbose=verbose)
        p.replace(local_list_path)
    return np.loadtxt(local_list_path, dtype="str")


def read_tile(
        lat: str, lon: str, resolution: int, directory: Path, available: list = None
    ):
    """
    Read a single tile as a dask array

    Data is accessed on demand
    """
    tile = _CopDEM_tile(lat, lon, resolution, directory)
    if available is not None and tile.tile_name not in available:
        tile.set_as_empty()
    return from_array(tile, meta=np.array([], tile.dtype))


def list_tiles() -> Tuple[List]:
    """Return list of latlon acronym"""
    lons = [f"W{w:03d}" for w in range(180, 0, -1)]
    lons.extend([f"E{e:03d}" for e in range(0, 180)])
    lats = [f"S{s:02d}" for s in range(90, 0, -1)]
    lats.extend([f"N{n:02d}" for n in range(0, 90)])
    return lats, lons


def generate_full_gsw_mosaic(
        directory: str | Path, resolution: int, verbose: bool = True
    ) -> xr.DataArray:
    """Build a lazy dask array representing the full global surface water mosaic"""

    # Determines list of latlon acronyms
    available = _list_available_tiles(directory, resolution, verbose)
    lats, lons = list_tiles()

    # concat the delayed dask objects for all tiles
    return concatenate([
        concatenate([
            read_tile(
                lat, lon,
                resolution=resolution,
                directory=directory,
                available=available,
            )
            for lat in lats
        ], axis=0)
        for lon in lons
    ], axis=1)
