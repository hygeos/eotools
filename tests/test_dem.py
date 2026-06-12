#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for digital elevation model (DEM) processors
"""

import pytest
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

from eotools.dem import GTOPO30, ZeroAltitude, CopernicusDEM
from core.geo.naming import names
from core.tests import conftest

# Shared test locations: (name, lat, lon, elev_min, elev_max)
# Used as pytest parametrize values — name becomes the test ID
KNOWN_ELEVATIONS = [
    ("lille", 50.63, 3.06, 10, 60),
    ("mont_blanc", 45.83, 6.86, 4000, 4900),
    ("bay_of_ushuaia", -54.86, -68.20, 0, 0),
]

# DEM processor parametrization: (id, class, kwargs_dict)
# CopernicusDEM is tested at both 30m and 90m resolutions
DEM_PROCESSORS = [
    ("gtopo30", GTOPO30, {}),
    ("copernicus_90m", CopernicusDEM, {"resolution": 90}),
    ("copernicus_30m", CopernicusDEM, {"resolution": 30}),
]


def _plot_dem_elevation(block, title, request, center_lat=None, center_lon=None):
    """Plot DEM altitude with a terrain-appropriate colormap and colorbar.

    Parameters
    ----------
    block : xr.Dataset
        Dataset containing 'altitude', 'latitude', and 'longitude'.
    title : str
        Title for the plot.
    request : pytest.FixtureRequest
        Used to save the figure to the test report.
    center_lat, center_lon : float or None
        If provided, plot a marker at the point of interest.
    """
    alt = block["altitude"].values
    lat = block[str(names.lat)].values
    lon = block[str(names.lon)].values

    # Determine valid elevation range (exclude ocean — ocean is 0)
    valid_mask = alt > 0
    if valid_mask.any():
        vmin, vmax = np.nanpercentile(alt[valid_mask], [2, 98])
    else:
        vmin, vmax = 0, 5000

    fig, ax = plt.subplots(figsize=(8, 6))
    # Use 'terrain' colormap — designed for elevation data
    im = ax.pcolormesh(
        lon, lat, alt,
        cmap="terrain",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    fig.colorbar(im, ax=ax, label="Altitude (m)")
    # Mark the point of interest
    if center_lat is not None and center_lon is not None:
        ax.plot(
            center_lon, center_lat,
            marker="^", markersize=10, color="red",
            markeredgecolor="white", markeredgewidth=1.2,
            zorder=5, label=f"({center_lat:.2f}°N, {center_lon:.2f}°E)",
        )
        ax.legend(loc="upper right", framealpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    conftest.savefig(request)


class TestZeroAltitude:
    """Tests for ZeroAltitude processor"""

    def test_init_altitude_process_block(self):
        """Test that ZeroAltitude sets altitude to zero"""
        processor = ZeroAltitude()

        # Create test dataset
        lat = xr.DataArray(
            np.array([45.0, 46.0, 47.0]), dims=[str(names.lat)], name=str(names.lat)
        )
        lon = xr.DataArray(
            np.array([5.0, 6.0, 7.0]), dims=[str(names.lon)], name=str(names.lon)
        )
        block = xr.Dataset({str(names.lat): lat, str(names.lon): lon})

        # Process
        processor.process_block(block)

        # Check that altitude was created and is zero
        assert "altitude" in block
        assert block["altitude"].attrs["units"] == "m"
        np.testing.assert_array_equal(block["altitude"].values, 0.0)
        assert block["altitude"].shape == lat.shape


class Test_DEM:
    """Parametrized tests for DEM processors.

    Tests GTOPO30 and CopernicusDEM (at both 30m and 90m resolutions) by:
    1. Processing a block over a known location
    2. Checking the center point elevation against expected bounds
    3. Producing a terrain-colored elevation plot
    """

    @pytest.mark.parametrize(
        "dem_id,dem_class,dem_kwargs", DEM_PROCESSORS, ids=[x[0] for x in DEM_PROCESSORS]
    )
    @pytest.mark.parametrize(
        "name,lat,lon,elev_min,elev_max", KNOWN_ELEVATIONS, ids=[x[0] for x in KNOWN_ELEVATIONS]
    )
    def test_process_block_and_elevation(
        self, request, dem_id, dem_class, dem_kwargs, name, lat, lon, elev_min, elev_max
    ):
        """Process a block over a known location, check center elevation, and plot.

        This test merges the process_block structure test with the known elevation
        check, producing a terrain-colored plot for visual inspection.
        """
        # Skip GTOPO30 for locations outside its coverage (56S - 60N)
        if dem_id == "gtopo30" and (lat < -56 or lat > 60):
            pytest.skip(f"GTOPO30 does not cover {name} (lat={lat}, outside 56S-60N)")

        # Build constructor kwargs — missing=0 means ocean returns 0
        init_kwargs = {"missing": 0, **dem_kwargs}

        # GTOPO30 requires lat/lon bounds at construction; CopernicusDEM does not
        if dem_class is GTOPO30:
            init_kwargs["lat"] = (lat - 0.75, lat + 0.75)
            init_kwargs["lon"] = (lon - 0.75, lon + 0.75)

        processor = dem_class(**init_kwargs)

        # Create test block: ~1.5 x ~1.5 degree grid centered on the location
        half = 0.75
        # Clamp latitude bounds to valid range
        lat_min = max(lat - half, -90)
        lat_max = min(lat + half, 90)
        n = 500  # grid resolution for a meaningful plot
        lat_arr = np.linspace(lat_min, lat_max, n)
        lon_arr = np.linspace(lon - half, lon + half, n)
        lat_grid, lon_grid = np.meshgrid(lat_arr, lon_arr, indexing="ij")
        ds = xr.Dataset(
            {str(names.lat): (["y", "x"], lat_grid), str(names.lon): (["y", "x"], lon_grid)}
        )

        # Process block
        processor.process_block(ds)

        # Check that altitude was created with correct shape
        assert "altitude" in ds, f"{dem_id}: altitude not created"
        assert ds["altitude"].shape == lat_grid.shape, f"{dem_id}: altitude shape mismatch"
        assert ds["altitude"].attrs.get("units") == "m", f"{dem_id}: altitude units missing"

        # Check center point elevation
        alt_values = ds["altitude"].values
        center_alt = float(alt_values[n // 2, n // 2])  # Center of n×n grid

        # check elevation is within expected bounds
        assert elev_min <= center_alt <= elev_max, (
            f"{dem_id} at {name}: center elevation {center_alt} not in [{elev_min}, {elev_max}]"
        )

        # Plot with terrain colormap
        title = f"{dem_id} — {name} ({lat:.2f}°N, {lon:.2f}°E), elev={center_alt:.0f}m"
        _plot_dem_elevation(ds, title, request, center_lat=lat, center_lon=lon)
