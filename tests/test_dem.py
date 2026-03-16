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

from eotools.dem import GTOPO30, InitAltitude
from core.geo.naming import names


class TestInitAltitude:
    """Tests for InitAltitude processor"""
    
    def test_init_altitude_process_block(self):
        """Test that InitAltitude sets altitude to zero"""
        processor = InitAltitude()
        
        # Create test dataset
        lat = xr.DataArray(
            np.array([45.0, 46.0, 47.0]),
            dims=[str(names.lat)],
            name=str(names.lat)
        )
        lon = xr.DataArray(
            np.array([5.0, 6.0, 7.0]),
            dims=[str(names.lon)],
            name=str(names.lon)
        )
        block = xr.Dataset({
            str(names.lat): lat,
            str(names.lon): lon
        })
        
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
    def mock_dem_dataset(self):
        """Create a mock DEM dataset"""
        lat = np.linspace(-56, 60, 100)
        lon = np.linspace(-180, 180, 200)
        
        # Create synthetic elevation data
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        # Simple elevation model: higher near equator, some noise
        elev = 1000 * np.exp(-((lat_grid/30)**2)) + 100 * np.random.rand(*lat_grid.shape)
        
        # Use 'lat' and 'lon' as dimension names to match xrcrop expectations
        ds = xr.Dataset(
            {
                "elev": (["lat", "lon"], elev)
            },
            coords={
                "lat": lat,
                "lon": lon
            }
        )
        return ds
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_gtopo30_init(self, temp_dir):
        """Test GTOPO30 initialization with custom directory"""
                    
        processor = GTOPO30(directory=str(temp_dir), missing=-999)
        
        assert processor.missing == -999
        assert hasattr(processor, 'dem')
    
    def test_gtopo30_method_linear(self, temp_dir):
        """Test GTOPO30 with linear interpolation method"""
            
        from core.interpolate import Linear
        processor = GTOPO30(directory=str(temp_dir), method='linear')
        
        assert processor.method == Linear
    
    def test_gtopo30_invalid_method(self, temp_dir):
        """Test GTOPO30 raises error with invalid interpolation method"""
        with pytest.raises(ValueError):
            GTOPO30(directory=str(temp_dir), method='invalid')
    
    def test_gtopo30_process_block(self, temp_dir):
        """Test GTOPO30 process_block method (basic structure test)"""
            
        processor = GTOPO30(directory=str(temp_dir))
        
        # Create test block with lat/lon
        lat = xr.DataArray(np.array([[45.0, 46.0, 47.0]]*3), dims=['y','x'])
        lon = xr.DataArray(np.array([[5.0, 6.0, 7.0]]*3), dims=['y','x'])
        block = xr.Dataset({
            str(names.lat): lat,
            str(names.lon): lon
        })
        
        processor.process_block(block)
        block.compute()