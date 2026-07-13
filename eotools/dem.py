from shutil import copy2
from tempfile import TemporaryDirectory

import xarray as xr
import numpy as np

from core.geo.naming import names
from core.network.download import download_url
from core.process.blockwise import BlockProcessor
from core.interpolate import Interpolator, Linear, Nearest
from core.tools import Var, xrcrop
from core import env, mdir
from core.files.fileutils import filegen

from typing import Any, Literal, Tuple
from xarray import Dataset
from pathlib import Path


class DEM:
    """
    Factory class that wraps the available DEM (Digital Elevation Model) processors.

    Creates the appropriate DEM BlockProcessor based on the `source` parameter:

    - `"zero"` → `ZeroAltitude` (all altitudes set to 0)
    - `"gtopo30"` → `GTOPO30` (30 arc-second global DEM, ~1km resolution)
    - `"copernicus"` → `CopernicusDEM` (90m or 30m Copernicus DEM)

    Args:
    -----
    ds : xr.Dataset, optional
        Level-1 dataset (used to extract lat/lon bounds for DEM loading).
        Required for `"gtopo30"` and `"copernicus"` unless `lat`/`lon` are provided.
    source : str, default `"zero"`
        Which DEM source to use: `"zero"`, `"gtopo30"`, or `"copernicus"`.
    **kwargs :
        Additional keyword arguments passed to the DEM constructor:

        - For `GTOPO30`: `lat`, `lon`, `directory`, `missing`, `method`, `verbose`
        - For `CopernicusDEM`: `lat`, `lon`, `directory`, `resolution`, `missing`, `verbose`

    Returns:
    --------
    BlockProcessor
        An instance of the selected DEM processor class.

    Examples:
    ---------
    >>> dem = DEM(ds, source="copernicus", resolution=30)
    >>> result = dem.map_blocks(ds)
    """

    def __new__(cls, ds: xr.Dataset | None = None, source: str = "zero", **kwargs) -> "BlockProcessor":
        source = source.lower()
        if source == "zero":
            return ZeroAltitude()
        elif source == "gtopo30":
            if ds is not None:
                return GTOPO30(ds, **kwargs)
            return GTOPO30(**kwargs)
        elif source == "copernicus":
            if ds is not None:
                return CopernicusDEM(ds, **kwargs)
            return CopernicusDEM(**kwargs)
        else:
            raise ValueError(
                f"Unknown DEM source: {source!r}. "
                f"Available sources: ['zero', 'gtopo30', 'copernicus']"
            )


class ZeroAltitude(BlockProcessor):
    """
    Bypasses DEM, by setting all altitudes to zero
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
        l1: xr.Dataset | None = None,
        lat: Tuple[float, float] | None = None,
        lon: Tuple[float, float] | None = None,
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
        url = "https://github.com/hygeos/eotools/releases/download/root/GTOPO30_DZ_MLUT.nc"
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
        Interpolator(
            self.dem,
            lat=self.method(str(names.lat)),
            lon=self.method(str(names.lon)),
        ).process_block(block)
        block["altitude"] = xr.where(block.elev >= 0, block.elev, self.missing)
        block["altitude"].attrs["units"] = "m"


def _copdem_url_prefix(resolution: Literal[30, 90]) -> str:
    """Return the address of Copernicus DEM API server based on resolution"""
    return f"https://copernicus-dem-{resolution}m.s3.eu-central-1.amazonaws.com"


def _copdem_tile_width(abs_lat_degree: int, resolution: int) -> int:
    """
    Return the number of columns in a Copernicus DEM tile for a given
    absolute latitude degree and resolution.

    The DEM uses variable longitude spacing to maintain near constant ground
    resolution at higher latitudes:

      | Latitude band | Columns (90m) | Columns (30m) |
      |---------------|---------------|---------------|
      | 0 - 49        | 1200          | 3600          |
      | 50 - 59       | 800           | 2400          |
      | 60 - 69       | 600           | 1800          |
      | 70 - 79       | 400           | 1200          |
      | 80 - 84       | 240           | 720           |
      | 85 - 89       | 120           | 360           |
    """
    base = 1200 * 90 // resolution  # rows per tile (also max cols at equator)
    if abs_lat_degree < 50:
        return base
    elif abs_lat_degree < 60:
        return base * 2 // 3
    elif abs_lat_degree < 70:
        return base // 2
    elif abs_lat_degree < 80:
        return base // 3
    elif abs_lat_degree < 85:
        return base // 5
    else:
        return base // 10


def _copdem_tile_name(row: int, col: int, resolution: int) -> str:
    """
    Build the Copernicus DEM tile filename stem (without .tif).

    Naming convention: Copernicus_DSM_COG_{res_arcsec}_{lat}{dir}_00_{lon}{dir}_00_DEM
    e.g. Copernicus_DSM_COG_30_N54_00_E009_00_DEM

    The name denotes the lower-left corner of the 1° tile.
    """
    arcsec = resolution // 3
    lat_dir = "N" if row >= 0 else "S"
    lon_dir = "E" if col >= 0 else "W"
    return f"Copernicus_DSM_COG_{arcsec}_{lat_dir}{abs(row):02d}_00_{lon_dir}{abs(col):03d}_00_DEM"


def _list_available_tiles(
        directory: Path, resolution: Literal[30, 90], verbose: bool = True
    ) -> set[str]:
    """
    Query and cache the list of available Copernicus DEM tile names.

    Returns a set of tile name stems (without .tif extension) for fast lookup.
    """
    list_filename = "tileList.txt"
    local_list_path = directory / f"tileList_{resolution}.txt"
    url = f"{_copdem_url_prefix(resolution)}/{list_filename}"

    @filegen(if_exists='skip')
    def download_tile_list(target: Path):
        with TemporaryDirectory() as tmpdir:
            tilelist = download_url(url, Path(tmpdir), if_exists="skip", verbose=verbose)
            copy2(tilelist, target)
    
    download_tile_list(local_list_path)

    return set(np.loadtxt(local_list_path, dtype="str"))


class CopernicusDEM(BlockProcessor):
    """
    Copernicus digital elevation model, 90m or 30m, with functions
    to get the altitude for a lat-lon grid and to download on demand DEM tiles
    via internet.

    Uses per-pixel tile lookup: for each (lat, lon) point, determines which
    1° tile covers that point, downloads the tile on demand, and indexes
    into it using the tile's actual dimensions.

    The DEM is a grid of tie points, not of pixels. The altitude is given for
    the degree line. There are 1200 rows per degree. The number of columns
    depends on the latitude band (fewer columns at higher latitudes, each
    spanning wider longitude).

    Args:
    -----
    l1: optional Xarray Dataset from an eoread reader (used to extract lat/lon bounds)
    lat: optional tuple (min, max) latitude bounds for pre-downloading tiles
    lon: optional tuple (min, max) longitude bounds for pre-downloading tiles
    directory: directory for tile storage
    resolution: 90 or 30, default 90
    missing: value to use for missing (ocean) elevation data (default: 0)
    verbose: with trace output
    """

    def __init__(
        self,
        l1: xr.Dataset | None = None,
        lat: Tuple[float, float] | None = None,
        lon: Tuple[float, float] | None = None,
        directory: str | Path | None = None,
        resolution: Literal[30, 90] = 90,
        missing: Any = 0,
        verbose: bool = True,
    ):
        if directory is None:
            directory = mdir(env.getdir("DIR_ANCILLARY") / "COPDEM")
        else:
            directory = Path(directory)

        self.resolution = resolution
        self.missing = missing
        self.verbose = verbose
        self.cache_directory = directory
        self.tile_height = int(1200 * 90 / resolution)

        # Cache for downloaded tiles: (row, col) -> numpy array
        self._tile_cache: dict[tuple[int, int], np.ndarray] = {}

        # Fetch list of available tiles
        self.available = _list_available_tiles(directory, resolution, verbose)
        if self.verbose:
            print(f"{len(self.available)} remote DEM tiles available")

        # Determine lat/lon bounds for pre-downloading tiles.
        # Priority: l1 > explicit lat/lon tuples.
        # Pre-downloading ensures concurrent processes won't interfere
        # with each other during process_block.
        if l1 is not None:
            lat_bounds = (
                float(l1[str(names.lat)].min()),
                float(l1[str(names.lat)].max()),
            )
            lon_bounds = (
                float(l1[str(names.lon)].min()),
                float(l1[str(names.lon)].max()),
            )
            self._pre_download_tiles(lat_bounds, lon_bounds)
        elif lat is not None and lon is not None:
            self._pre_download_tiles(lat, lon)

    def input_vars(self) -> list[Var]:
        return [names.lat, names.lon]

    def created_vars(self) -> list[Var]:
        return [Var("altitude", dims_like=str(names.lat), attrs={"units": "m"})]
    
    def auto_template(self) -> bool:
        return True

    def _pre_download_tiles(self, lat: Tuple[float, float], lon: Tuple[float, float]) -> None:
        """
        Pre-download all DEM tiles that cover the given lat/lon bounds.

        This ensures that concurrent processes calling process_block won't
        interfere with each other during tile downloads.
        """
        # Determine the set of tile rows and columns that cover the bounds
        half_pixel_height = 0.5 / self.tile_height
        row_min = int(np.floor(lat[0] - half_pixel_height))
        row_max = int(np.floor(lat[1] - half_pixel_height))

        # Tile width varies by latitude; compute for each row
        for row in range(row_min, row_max + 1):
            tw = _copdem_tile_width(abs(row), self.resolution)
            half_pixel_width = 0.5 / tw
            col_min = int(np.floor((lon[0] + half_pixel_width + 180.0) % 360.0 - 180.0))
            col_max = int(np.floor((lon[1] + half_pixel_width + 180.0) % 360.0 - 180.0))

            for col in range(col_min, col_max + 1):
                self._get_tile(row, col)

    def _get_tile(self, row: int, col: int) -> np.ndarray | None:
        """
        Get or download a single DEM tile.

        Returns the tile data as a 2D numpy array (rows x cols), or None
        if the tile is not available.
        """
        if (row, col) in self._tile_cache:
            return self._tile_cache[(row, col)]

        tile_name = _copdem_tile_name(row, col, self.resolution)

        # Check availability
        if tile_name not in self.available:
            return None

        # Build download URL
        prefix = _copdem_url_prefix(self.resolution)
        url = f"{prefix}/{tile_name}/{tile_name}.tif"
        local_path = self.cache_directory / f"{tile_name}.tif"

        # Download if needed
        if not local_path.exists():
            download_url(url, self.cache_directory, if_exists="skip", verbose=False)

        # Read tile data
        if self.verbose:
            print(f"reading DEM tile {local_path}")
        with xr.open_dataset(str(local_path), engine="rasterio") as ds:
            dem = ds.band_data.values[0]

        # DEM rows are ordered from North to South (first line is North-most).
        # The formula (row + 1) - lat already accounts for this, so no flip needed.
        self._tile_cache[(row, col)] = dem
        return dem

    def _get_altitude(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Get altitude for arrays of lat/lon coordinates.

        Follows the reference implementation: for each point, determines
        which tile covers it, then indexes into that tile using the
        tile's actual dimensions.
        """
        altitude = np.empty(lat.shape, dtype=np.float32)
        altitude[:] = np.nan

        # Determine tile row for each point
        # (A pixel center less than half a pixel above the degree line is covered by this tile)
        half_pixel_height = 0.5 / self.tile_height
        rows = np.floor(lat - half_pixel_height).astype(np.int32)

        # Determine tile width for each point (varies by latitude band)
        tile_width = np.empty(lat.shape, dtype=np.int32)
        tile_width[:] = self.tile_height
        tile_width[(rows >= 50) | (rows <= -50)] = self.tile_height * 2 // 3
        tile_width[(rows >= 60) | (rows <= -60)] = self.tile_height // 2
        tile_width[(rows >= 70) | (rows <= -70)] = self.tile_height // 3
        tile_width[(rows >= 80) | (rows <= -80)] = self.tile_height // 5
        tile_width[(rows >= 85) | (rows <= -85)] = self.tile_height // 10

        # Determine tile col for each point
        # (A pixel center less than half a pixel left of the degree line is covered by this tile)
        half_pixel_width = 0.5 / tile_width
        cols = np.floor((lon + half_pixel_width + 180.0) % 360.0 - 180.0).astype(np.int32)

        # Encode as unique bin index per tile
        bin_index = (rows + 90) * 360 + cols + 180

        # Process each unique tile
        bin_set = np.unique(bin_index)
        for bin_val in bin_set:
            row = bin_val // 360 - 90
            col = bin_val % 360 - 180

            # Get tile data
            dem = self._get_tile(row, col)
            if dem is None:
                continue

            # Mask for points in this tile
            is_inside_tile = bin_index == bin_val
            if not is_inside_tile.any():
                continue

            # Compute within-tile indices
            dem_row = (((row + 1) - lat[is_inside_tile] + half_pixel_height) * self.tile_height).astype(np.int32)
            dem_col = (((lon[is_inside_tile] - col + half_pixel_width[is_inside_tile]) % 360.0) * tile_width[is_inside_tile]).astype(np.int32)

            # Clamp indices to valid range
            dem_row = np.clip(dem_row, 0, dem.shape[0] - 1)
            dem_col = np.clip(dem_col, 0, dem.shape[1] - 1)

            altitude[is_inside_tile] = dem[dem_row, dem_col]

        # Fill missing values
        if self.missing is not None:
            altitude[np.isnan(altitude)] = self.missing

        return altitude

    def process_block(self, block: xr.Dataset) -> None:
        lat = np.asarray(block[str(names.lat)])
        lon = np.asarray(block[str(names.lon)])

        altitude = self._get_altitude(lat, lon)

        block["altitude"] = xr.DataArray(
            altitude,
            dims=block[str(names.lat)].dims,
            attrs={"units": "m"},
        )
