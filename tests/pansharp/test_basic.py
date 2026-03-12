import pytest
import numpy as np
import xarray as xr
from eotools.pansharp.basic import Brovey, valid_ms_vars
from core.geo.naming import names, Var


class TestBrovey:
    """Test suite for the Brovey pansharpening class."""
    
    @pytest.fixture
    def spatial_mapping(self):
        """Spatial mapping for tests."""
        return {str(names.rows): 'pan_y', str(names.columns): 'pan_x'}
    
    def test_initialization_without_weights(self, spatial_mapping):
        """Test Brovey initialization with default weights (None)."""
        brovey = Brovey(var_ms=valid_ms_vars[0], panchro="pan", spatial_mapping=spatial_mapping)
        assert brovey.var_ms == valid_ms_vars[0]
        assert brovey.panchro == "pan"
        assert brovey.spatial_mapping == spatial_mapping
        assert brovey.weights is None
    
    def test_initialization_with_weights(self, spatial_mapping):
        """Test Brovey initialization with custom weights."""
        weights = [0.25, 0.5, 0.25]
        brovey = Brovey(var_ms=valid_ms_vars[1], panchro="pan", spatial_mapping=spatial_mapping, weights=weights)
        assert brovey.var_ms == valid_ms_vars[1]
        assert brovey.panchro == "pan"
        assert brovey.spatial_mapping == spatial_mapping
        assert brovey.weights == weights
    
    def test_input_vars_valid(self, spatial_mapping):
        """Test input_vars returns correct variables for valid var_ms."""
        brovey = Brovey(var_ms=valid_ms_vars[0], panchro="pan", spatial_mapping=spatial_mapping)
        
        result = brovey.input_vars()
        
        assert len(result) == 2
        assert isinstance(result[0], Var)
        assert isinstance(result[1], Var)
    
    def test_input_vars_invalid_raises_assertion(self, spatial_mapping):
        """Test input_vars raises AssertionError for invalid var_ms."""
        brovey = Brovey(var_ms="invalid_var", panchro="pan", spatial_mapping=spatial_mapping)
        
        with pytest.raises(AssertionError):
            brovey.input_vars()
    
    def test_created_vars(self, spatial_mapping):
        """Test created_vars returns upsample variable."""
        brovey = Brovey(var_ms=valid_ms_vars[0], panchro="pan", spatial_mapping=spatial_mapping)
        result = brovey.created_vars()
        
        assert len(result) == 1
        assert isinstance(result[0], Var)
    
    def test_process_block_without_weights(self, spatial_mapping):
        """Test process_block with default weights (mean calculation)."""
        var_ms = valid_ms_vars[0]
        brovey = Brovey(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
        # Create real data with proper dimensions
        pan_data = xr.DataArray(
            np.ones((10, 10)) * 100,
            dims=['pan_y', 'pan_x']
        )
        ms_data = xr.DataArray(
            np.ones((5, 5, 3)) * 50,
            dims=[str(names.rows), str(names.columns), str(names.bands)]
        )
        
        block = {
            "pan": pan_data,
            var_ms: ms_data
        }
        
        brovey.process_block(block)
        result = block['upsample']
        
        # Result should have upsampled dimensions (5x5 -> 10x10)
        assert result is not None
        # Should have panchromatic dimension names after renaming
        assert 'pan_y' in result.dims
        assert 'pan_x' in result.dims
        assert str(names.bands) in result.dims
        # Check sizes
        assert result.sizes['pan_y'] == 10
        assert result.sizes['pan_x'] == 10
        assert result.sizes[str(names.bands)] == 3
    
    def test_process_block_with_weights(self, spatial_mapping):
        """Test process_block with custom weights."""
        var_ms = valid_ms_vars[0]
        weights = [0.2, 0.5, 0.3]
        brovey = Brovey(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping, weights=weights)
        
        # Create real data with proper dimensions
        pan_data = xr.DataArray(
            np.ones((10, 10)) * 100,
            dims=['pan_y', 'pan_x']
        )
        ms_data = xr.DataArray(
            np.ones((5, 5, 3)) * 50,
            dims=[str(names.rows), str(names.columns), str(names.bands)]
        )
        
        block = {
            "pan": pan_data,
            var_ms: ms_data
        }
        
        brovey.process_block(block)
        result = block['upsample']
        
        # Result should have upsampled dimensions
        assert result is not None
        assert 'pan_y' in result.dims
        assert 'pan_x' in result.dims
        assert result.sizes['pan_y'] == 10
        assert result.sizes['pan_x'] == 10
        assert result.sizes[str(names.bands)] == 3
    
    def test_process_block_spatial_upsampling(self, spatial_mapping):
        """Test that spatial resolution is upsampled by factor of 2."""
        var_ms = valid_ms_vars[0]
        brovey = Brovey(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
        # Create simple test data with known values
        pan_data = xr.DataArray(
            np.ones((10, 10)) * 100,
            dims=['pan_y', 'pan_x']
        )
        
        # Create multispectral data with smaller spatial dimensions
        ms_data = xr.DataArray(
            np.ones((5, 5, 3)) * 50,
            dims=[str(names.rows), str(names.columns), str(names.bands)]
        )
        
        block = {
            "pan": pan_data,
            var_ms: ms_data
        }
        
        brovey.process_block(block)
        result = block['upsample']
        
        # After upsampling 5x5 should become 10x10
        assert result.sizes['pan_y'] == 10
        assert result.sizes['pan_x'] == 10
        assert result.sizes[str(names.bands)] == 3
    
    def test_process_block_brovey_transformation(self, spatial_mapping):
        """Test the actual Brovey transformation calculation."""
        var_ms = valid_ms_vars[0]
        brovey = Brovey(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
        # Create test data with known values
        pan_data = xr.DataArray(
            np.ones((10, 10)) * 200,
            dims=['pan_y', 'pan_x']
        )
        ms_data = xr.DataArray(
            np.ones((5, 5, 3)) * 50,
            dims=[str(names.rows), str(names.columns), str(names.bands)]
        )
        
        block = {
            "pan": pan_data,
            var_ms: ms_data
        }
        
        brovey.process_block(block)
        result = block['upsample']
        
        # The ratio should be 200/50 = 4 (pan / mean of bands)
        # So pansharp should be approximately 50 * 4 = 200
        assert result is not None
        assert np.all(result.values > 0)
        # Check that the transformation is applied correctly
        # With equal weights: pseudopan = mean(50, 50, 50) = 50
        # ratio = 200/50 = 4
        # pansharp = 50 * 4 = 200
        assert np.allclose(result.values, 200)
    
    def test_process_block_with_weighted_transformation(self, spatial_mapping):
        """Test Brovey transformation with custom weights."""
        var_ms = valid_ms_vars[0]
        weights = [0.5, 0.3, 0.2]
        brovey = Brovey(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping, weights=weights)
        
        # Create test data with different band values
        pan_data = xr.DataArray(
            np.ones((10, 10)) * 100,
            dims=['pan_y', 'pan_x']
        )
        # Different values for each band
        ms_array = np.stack([
            np.ones((5, 5)) * 40,
            np.ones((5, 5)) * 60,
            np.ones((5, 5)) * 80
        ], axis=-1)
        ms_data = xr.DataArray(
            ms_array,
            dims=[str(names.rows), str(names.columns), str(names.bands)]
        )
        
        block = {
            "pan": pan_data,
            var_ms: ms_data
        }
        
        brovey.process_block(block)
        result = block['upsample']
        
        # Weighted pseudopan = 0.5*40 + 0.3*60 + 0.2*80 = 20 + 18 + 16 = 54
        # ratio = 100/54 ≈ 1.852
        # pansharp bands should be: [40*1.852, 60*1.852, 80*1.852]
        assert result is not None
        assert np.all(result.values > 0)
    
    def test_valid_ms_vars_list(self):
        """Test that valid_ms_vars contains expected string values."""
        assert isinstance(valid_ms_vars, list)
        assert len(valid_ms_vars) == 3
        for var in valid_ms_vars:
            assert isinstance(var, str)
    
    @pytest.mark.parametrize("var_index", [0, 1, 2])
    def test_different_var_ms_types(self, var_index, spatial_mapping):
        """Test Brovey works with different valid var_ms types."""
        var_ms = valid_ms_vars[var_index]
        brovey = Brovey(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        assert brovey.var_ms == var_ms
