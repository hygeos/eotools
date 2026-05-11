#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for digital elevation model (DEM) processors
"""

import pytest
import xarray as xr
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from eotools.dem import GTOPO30, InitAltitude, CopernicusDEM
from core.geo.naming import names
from core.tests import conftest


class TestInitAltitude:
    """Tests for InitAltitude processor"""

    def test_init_altitude_process_block(self):
        """Test that InitAltitude sets altitude to zero"""
        processor = InitAltitude()

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


class TestGTOPO30:
    """Tests for GTOPO30 processor"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_gtopo30_init(self, temp_dir):
        """Test GTOPO30 initialization with custom directory"""

        processor = GTOPO30(
            lat=(45.0, 47.0), lon=(5.0, 7.0), directory=str(temp_dir), missing=-999
        )

        assert processor.missing == -999
        assert hasattr(processor, "dem")

    def test_gtopo30_init_with_l1(self, temp_dir):
        """Test GTOPO30 initialization with custom directory"""

        # Create test block with lat/lon
        lat = xr.DataArray(np.array([[45.0, 46.0, 47.0]] * 3), dims=["y", "x"])
        lon = xr.DataArray(np.array([[5.0, 6.0, 7.0]] * 3), dims=["y", "x"])
        l1 = xr.Dataset({str(names.lat): lat, str(names.lon): lon})

        processor = GTOPO30(l1=l1, directory=str(temp_dir), missing=-999)

        assert processor.missing == -999
        assert hasattr(processor, "dem")

    def test_gtopo30_method_linear(self, temp_dir):
        """Test GTOPO30 with linear interpolation method"""

        from core.interpolate import Linear

        processor = GTOPO30(
            lat=(45.0, 47.0), lon=(5.0, 7.0), directory=str(temp_dir), method="linear"
        )

        assert processor.method == Linear

    def test_gtopo30_invalid_method(self, temp_dir):
        """Test GTOPO30 raises error with invalid interpolation method"""
        with pytest.raises(ValueError):
            GTOPO30(
                lat=(45.0, 47.0),
                lon=(5.0, 7.0),
                directory=str(temp_dir),
                method="invalid",
            )

    def test_gtopo30_process_block(self, temp_dir, request):
        """Test GTOPO30 process_block method (basic structure test)"""

        processor = GTOPO30(lat=(45.0, 47.0), lon=(5.0, 7.0), directory=str(temp_dir))

        # Create test block with lat/lon
        lat = xr.DataArray(np.array([[45.0, 46.0, 47.0]] * 3), dims=["y", "x"])
        lon = xr.DataArray(np.array([[5.0, 6.0, 7.0]] * 3), dims=["y", "x"])
        block = xr.Dataset({str(names.lat): lat, str(names.lon): lon})

        # Trigger computation
        processor.process_block(block)
        block = block.compute()

        # Plot results
        processor.dem.elev.plot.imshow()
        conftest.savefig(request)

    def test_gtopo30_partial_initialise(self, temp_dir):
        """Test GTOPO30 constructor without latlon constraints"""
        # Try to process without croping DEM raster
        with pytest.raises(ValueError):
            GTOPO30(directory=str(temp_dir))


class TestCopernicusDEM:
    """Tests for CopernicusDEM processor"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_CopernicusDEM_init(self, temp_dir):
        """Test CopernicusDEM initialization with custom directory"""

        processor = CopernicusDEM(
            lat=(46.5, 47.5), lon=(6.5, 7.5), directory=str(temp_dir), missing=-999
        )

        assert processor.missing == -999
        assert hasattr(processor, "dem")

    def test_CopernicusDEM_init_with_l1(self, temp_dir):
        """Test CopernicusDEM initialization with custom directory"""

        # Create test block with lat/lon
        lat = xr.DataArray(np.array([[46.5, 47.0, 47.5]] * 3), dims=["y", "x"])
        lon = xr.DataArray(np.array([[6.5, 7.0, 7.5]] * 3), dims=["y", "x"])
        l1 = xr.Dataset({str(names.lat): lat, str(names.lon): lon})

        processor = CopernicusDEM(l1=l1, directory=str(temp_dir), missing=-999)

        assert processor.missing == -999
        assert hasattr(processor, "dem")

    def test_CopernicusDEM_method_linear(self, temp_dir):
        """Test CopernicusDEM with linear interpolation method"""

        from core.interpolate import Linear

        processor = CopernicusDEM(
            lat=(46.5, 47.5), lon=(6.5, 7.5), directory=str(temp_dir), method="linear"
        )

        assert processor.method == Linear

    def test_CopernicusDEM_invalid_method(self, temp_dir):
        """Test CopernicusDEM raises error with invalid interpolation method"""
        with pytest.raises(ValueError):
            CopernicusDEM(
                lat=(46.5, 47.5),
                lon=(6.5, 7.5),
                directory=str(temp_dir),
                method="invalid",
            )

    def test_CopernicusDEM_process_block(self, temp_dir, request):
        """Test CopernicusDEM process_block method (basic structure test)"""

        processor = CopernicusDEM(
            lat=(46.5, 47.5), lon=(6.5, 7.5), directory=str(temp_dir)
        )

        # Create test block with lat/lon
        lat = xr.DataArray(np.array([[46.5, 47.0, 47.5]] * 3), dims=["y", "x"])
        lon = xr.DataArray(np.array([[6.5, 7.0, 7.5]] * 3), dims=["y", "x"])
        block = xr.Dataset({str(names.lat): lat, str(names.lon): lon})

        # Trigger computation
        processor.process_block(block)
        block.compute()

        # Plot results
        processor.dem.plot.imshow()
        conftest.savefig(request)

    def test_CopernicusDEM_partial_initialise(self, temp_dir):
        """Test CopernicusDEM constructor without latlon constraints"""
        # Try to process without croping DEM raster
        with pytest.raises(ValueError):
            CopernicusDEM(directory=str(temp_dir))
