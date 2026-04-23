#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for Global Surface Water (GSW) processor
"""

import pytest
import xarray as xr
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from dask import array as da

from eotools.water import GSW, read_tile, list_tiles, _GSW_tile, _url_tile
from core.geo.naming import names


class TestGSWUtilities:
    """Tests for GSW utility functions"""
    
    def test_url_tile_format(self):
        """Test that _url_tile generates correct URL format"""
        tile_name = "10E_20N"
        url = _url_tile(tile_name)
        
        expected_base = "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/"
        assert url.startswith(expected_base)
        assert tile_name in url
        assert url.endswith(".tif")
    
    def test_list_tiles_structure(self):
        """Test that list_tiles returns correct structure"""
        lats, lons = list_tiles()
        
        # Check that we have lists
        assert isinstance(lats, list)
        assert isinstance(lons, list)
        
        # Check expected number of tiles
        # Lons: 180W to 170W (18) + 0E to 170E (18) = 36 total
        # Lats: 50S to 10S (5) + 0N to 80N (9) = 14 total
        assert len(lons) == 36
        assert len(lats) == 14
    
    def test_list_tiles_longitude_format(self):
        """Test longitude tile naming format"""
        lats, lons = list_tiles()
        
        # Check W longitudes
        assert "180W" in lons
        assert "10W" in lons
        
        # Check E longitudes  
        assert "0E" in lons
        assert "170E" in lons
        
        # Check ordering (W then E)
        w_indices = [i for i, lon in enumerate(lons) if "W" in lon]
        e_indices = [i for i, lon in enumerate(lons) if "E" in lon]
        assert max(w_indices) < min(e_indices)
    
    def test_list_tiles_latitude_format(self):
        """Test latitude tile naming format"""
        lats, lons = list_tiles()
        
        # Check S latitudes
        assert "50S" in lats
        assert "10S" in lats
        
        # Check N latitudes
        assert "0N" in lats
        assert "80N" in lats
        
        # Check ordering (S then N)
        s_indices = [i for i, lat in enumerate(lats) if "S" in lat]
        n_indices = [i for i, lat in enumerate(lats) if "N" in lat]
        assert max(s_indices) < min(n_indices)
    
    def test_gsw_coordinate_conversion(self):
        """Test coordinate conversion logic used in GSW.__init__"""
        lats, lons = list_tiles()
        
        # Test the coordinate conversion logic from GSW.__init__
        lat_coords = [int(l[:-1]) if 'N' in l else -int(l[:-1]) for l in lats[::-1]]
        lon_coords = [int(l[:-1]) if 'E' in l else -int(l[:-1]) for l in lons]
        
        # Check some expected values
        assert 80 in lat_coords  # 80N
        assert -50 in lat_coords  # 50S
        assert 170 in lon_coords  # 170E
        assert -180 in lon_coords  # 180W
        
        # Verify ordering (N to S for reversed lats)
        assert lat_coords[0] > lat_coords[-1]  # First should be north, last should be south
        
        # Verify length
        assert len(lat_coords) == 14
        assert len(lon_coords) == 36


class TestGSWTile:
    """Tests for _GSW_tile class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_gsw_tile_init(self, temp_dir):
        """Test _GSW_tile initialization"""
        tile = _GSW_tile("10E_20N", agg=8, directory=temp_dir)
        
        assert tile.agg == 8
        assert tile.dtype == 'uint8'
        assert "v1_4_2021" in tile.tile_name
        assert tile.shape == (40000/8, 40000/8)
        assert tile.directory == temp_dir
    
    def test_gsw_tile_shape_calculation(self, temp_dir):
        """Test that tile shape is calculated correctly based on aggregation"""
        tile_agg1 = _GSW_tile("10E_20N", agg=1, directory=temp_dir)
        tile_agg4 = _GSW_tile("10E_20N", agg=4, directory=temp_dir)
        tile_agg8 = _GSW_tile("10E_20N", agg=8, directory=temp_dir)
        
        assert tile_agg1.shape == (40000, 40000)
        assert tile_agg4.shape == (40000/4, 40000/4)
        assert tile_agg8.shape == (40000/8, 40000/8)
    
    def test_gsw_tile_directory_not_exists(self):
        """Test that _GSW_tile raises error if directory doesn't exist"""
        non_existent = Path("/tmp/this_should_not_exist_234567890")
        
        with pytest.raises(IOError, match="does not exist"):
            _GSW_tile("10E_20N", agg=8, directory=non_existent)
    
    def test_gsw_tile_has_directory_attribute(self, temp_dir):
        """Test that _GSW_tile stores directory attribute (new in updated version)"""
        tile = _GSW_tile("10E_20N", agg=8, directory=temp_dir)
        assert hasattr(tile, 'directory')
        assert tile.directory == temp_dir
    
    def test_gsw_tile_has_fetch_method(self, temp_dir):
        """Test that _GSW_tile has fetch_gsw_tile method"""
        tile = _GSW_tile("10E_20N", agg=8, directory=temp_dir)
        assert hasattr(tile, 'fetch_gsw_tile')
        assert callable(tile.fetch_gsw_tile)
    
    def test_gsw_tile_getitem_structure(self, temp_dir):
        """Test that _GSW_tile.__getitem__ exists for dask array protocol"""
        tile = _GSW_tile("10E_20N", agg=8, directory=temp_dir)
        assert hasattr(tile, '__getitem__')
        assert callable(tile.__getitem__)


class TestReadTile:
    """Tests for read_tile function"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_read_tile_returns_dask_array(self, temp_dir):
        """Test that read_tile returns a dask array"""
        result = read_tile("10E_20N", agg=8, directory=temp_dir)
        
        assert isinstance(result, da.Array)
        assert result.dtype == np.uint8
    
    def test_read_tile_shape(self, temp_dir):
        """Test that read_tile creates array with correct shape"""
        result = read_tile("10E_20N", agg=8, directory=temp_dir)
        
        # Shape should match _GSW_tile shape
        assert result.shape == (40000/8, 40000/8)
    
    @pytest.mark.parametrize('tilename', ['10W_50S','10W_10N'])
    def test_read_tile_compute(self, temp_dir, tilename):
        """Test that read_tile creates array with correct shape"""
        tile = read_tile(tilename, agg=8, directory=temp_dir).compute()
        search = list(Path(temp_dir).glob(f'occurrence_{tilename}*_{8}.nc'))
        assert len(search) == 1
        


class TestGSWCompute:
    """Tests for GSW compute functionality with workarounds"""
    
    def test_GSW_init(self, tmpdir):
        """
        Test that xarray DataArray works with dask arrays (validates GSW approach)
        """
        GSW(lat=(45.0, 47.0), lon=(5.0, 7.0), directory=tmpdir)
    
    def test_GSW_compute(self, tmpdir):
        """
        Test that GSW processor can process a block with lat/lon coordinates
        """
        # Create a test dataset with latitude and longitude coordinates
        # Use coordinates in the GSW data range (50S to 80N, 180W to 170E)
        lat = xr.DataArray(np.array([[45.0, 46.0, 47.0]]*3).T, dims=['y','x'])
        lon = xr.DataArray(np.array([[5.0, 6.0, 7.0]]*3), dims=['y','x'])
        ds = xr.Dataset({str(names.lat): lat, str(names.lon): lon})
        
        # Process block using GSW
        gsw_processor = GSW(lat=(45.0, 47.0), lon=(5.0, 7.0), directory=tmpdir)
        ds = gsw_processor.map_blocks(ds.chunk(-1))
        ds.compute()
        
        # Verify that water variable was added to the block
        assert "water" in ds
        assert ds["water"].shape == lat.shape
        assert "units" in ds["water"].attrs
        assert ds["water"].attrs["units"] == "%"
    
    def test_GSW_compute_chunked(self, tmpdir):
        """
        Test that GSW processor can process chunked data with multiple chunks
        """
        # Create a larger dataset that will be split into multiple chunks
        lat_vals = np.linspace(45.0, 47.0, 10)
        lon_vals = np.linspace(5.0, 7.0, 10)
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
        
        lat = xr.DataArray(lat_grid, dims=['y', 'x'])
        lon = xr.DataArray(lon_grid, dims=['y', 'x'])
        ds = xr.Dataset({str(names.lat): lat, str(names.lon): lon})
        
        # Process block using GSW with small chunks to ensure multiple chunks
        gsw_processor = GSW(lat=(45.0, 47.0), lon=(5.0, 7.0), directory=tmpdir, agg=16)
        ds_chunked = ds.chunk({'y': 3, 'x': 3})
        result = gsw_processor.map_blocks(ds_chunked)
        result = result.compute()
        
        # Verify that water variable was added and has correct shape
        assert "water" in result
        assert result["water"].shape == lat.shape
        assert "units" in result["water"].attrs
        assert result["water"].attrs["units"] == "%"