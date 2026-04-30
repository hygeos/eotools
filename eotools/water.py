#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read global surface water from https://global-surface-water.appspot.com/

https://doi.org/10.1038/nature20584

Example:
-------

>>> gsw = GSW(agg=8)
Create water mask
>>> mask = gsw.sel(latitude=lat, longitude=lon, method='nearest') > 50
"""

import xarray as xr
import numpy as np

from pathlib import Path
from dask import array as da
from typing import Tuple, List

from core import env
from core.geo.naming import names
from core.tools import drop_unused_dims, Var, xrcrop
from core.interpolate import Nearest, Interpolator
from core.process.blockwise import BlockProcessor
from core.network.download import download_url
from core.save import to_netcdf
from core.files import mdir


def _url_tile(tile_name):
    url = "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{}.tif"
    return url.format(tile_name)


class GSW(BlockProcessor):
    """
    Global surface water reader

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
        agg: int = 8,
    ):

        if directory is None:
            directory = mdir(env.getdir("DIR_ANCILLARY") / "GSW")
        else:
            directory = Path(directory)

        # create GSW mosaic
        gsw = generate_full_gsw_mosaic(directory, agg)

        # FIXME: problem with coordinates
        dims = ("lat", "lon")
        step, off = 2.5e-4 * agg, 125e-6
        coords = dict(
            lon=np.linspace(
                off - 180, step * (gsw.shape[1] - 1) + off - 180, gsw.shape[1]
            ),
            lat=-np.linspace(
                off - 80, step * (gsw.shape[0] - 1) + off - 80, gsw.shape[0]
            )[::-1],
        )
        raster = xr.DataArray(gsw, name="occurrence", dims=dims, coords=coords)

        # Crop it to avoid loading all the raster
        if l1:
            self.water = xrcrop(raster, lat=l1[str(names.lat)], lon=l1[str(names.lon)])

        elif lat or lon:
            assert lat and lon, "Latitude and longitude constraints should be provided"
            dims = (names.rows, names.columns)
            lon = xr.DataArray([lon] * 2, dims=(names.rows, names.columns))
            lat = xr.DataArray([lat] * 2, dims=(names.columns, names.rows))
            self.water = xrcrop(raster, lat=lat, lon=lon)

        # Download needed GSW tiles
        self.water.compute(scheduler="synchronous")

    def input_vars(self) -> list[Var]:
        return [names.lat, names.lon]

    def created_vars(self) -> list[Var]:
        return [
            Var("water", dtype="uint8", attrs={"units": "%"}, dims_like=str(names.lat))
        ]

    def process_block(self, block: xr.Dataset) -> None:
        assert hasattr(self, "water"), "Provide LatLon constraints in constructor"
        params = dict(lat=Nearest(str(names.lat)), lon=Nearest(str(names.lon)))
        Interpolator(self.water.to_dataset(), **params).process_block(block)
        block["water"] = xr.where(block.occurrence >= 0, block.occurrence, 0)


class _GSW_tile:
    convert_missing_data = True

    def __init__(self, tile_name: str, agg: int, directory: Path):

        N = 40000 / agg
        self.shape = (N, N)
        self.dtype = "uint8"
        self.tile_name = tile_name + "v1_4_2021"
        self.directory = directory
        self.agg = agg

        if not directory.exists():
            raise IOError(
                f"Directory {directory} does not exist. "
                "It will be used to store GSW tiles. "
                "Please create it or link it first."
            )

    def __getitem__(self, key):
        return self.fetch_gsw_tile()[key].compute(scheduler="sync").values

    def fetch_gsw_tile(self) -> xr.DataArray:
        """
        Read remote file and returns its content as a numpy array
        """

        # Download tiles
        url = _url_tile(self.tile_name)
        netcdf_path = self._get_nc_path()

        # Branching to prevent memory usage peak
        if not netcdf_path.exists():
            # Download and read geotiff data
            p = download_url(url, self.directory, if_exists="skip", verbose=False)
            data = xr.open_dataarray(p, engine="rasterio").squeeze()
            data = data.isel(y=slice(None, None, -1))  # Flip values along y-axis
            data = data.rename(x=str(names.columns), y=str(names.rows))
            data = drop_unused_dims(data)
            data.name = "occurrence"

            # Aggregate data
            data = self._aggregate(data, self.agg)
            data.attrs["aggregation factor"] = str(self.agg)
            data.attrs["source_file"] = _url_tile(self.tile_name)

            # Save into netcdf to limit memory usage next time
            to_netcdf(data.to_dataset(), filename=netcdf_path, verbose=False)

        else:
            # Read netcdf file
            data = xr.open_dataarray(netcdf_path, engine="h5netcdf")

        # Fill missing values
        if self.convert_missing_data:
            val_nodata = 255
            data = data.where(
                data != val_nodata, 100
            )  # fill invalid data (assume water)

        return data.astype(self.dtype)

    def _get_nc_path(self) -> Path:
        """Determine the path of netcdf file containing the appropriate tile"""
        return self.directory / f"occurrence_{self.tile_name}_{self.agg}.nc"

    @staticmethod
    def _aggregate(A: xr.DataArray, agg: int = 1) -> xr.DataArray:
        """Aggregate array `A` by a factor `agg`"""
        assert agg > 0, f"aggregation factor should be positive, got {agg}"
        if agg == 1:
            return A

        assert (agg & (agg - 1)) == 0, f"agg should be a power of 2 ({agg})"
        A = A.chunk({str(names.columns): 5000, str(names.rows): 5000})
        group = A.coarsen(
            {str(names.columns): agg, str(names.rows): agg}, boundary="trim"
        )
        return group.mean().astype("uint8")


def read_tile(tile_name: str, agg: int, directory: Path) -> xr.DataArray:
    """
    Read a single tile as a dask array

    Data is accessed on demand
    """
    tile = _GSW_tile(tile_name, agg, directory)
    return da.from_array(tile, meta=np.array([], tile.dtype))


def list_tiles() -> Tuple[List]:
    """Return list of latlon acronym"""
    lons = [str(w) + "W" for w in range(180, 0, -10)]
    lons.extend([str(e) + "E" for e in range(0, 180, 10)])
    lats = [str(s) + "S" for s in range(50, 0, -10)]
    lats.extend([str(n) + "N" for n in range(0, 90, 10)])

    return lats, lons


def generate_full_gsw_mosaic(directory: str | Path, agg: int) -> xr.DataArray:
    """Build a lazy dask array representing the full global surface water mosaic"""

    # Determines list of latlon acronyms
    lats, lons = list_tiles()

    # concat the delayed dask objects for all tiles
    return da.concatenate(
        [
            da.concatenate(
                [read_tile(f"{lon}_{lat}", agg, directory) for lat in lats], axis=0
            )
            for lon in lons
        ],
        axis=1,
    )
