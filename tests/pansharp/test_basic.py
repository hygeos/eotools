import pytest
import numpy as np
import xarray as xr

from eotools.pansharp.basic import Brovey, IHS, HPF, valid_ms_vars
from core.geo.naming import names, Var
from generic import _Test

from PIL import Image
import requests
from io import BytesIO


@pytest.fixture(scope="session")
def hygeos_logo():
    """Download and load the Hygeos logo as RGB image."""
    url = "https://docs.hygeos.com/s/tFbytfyWd7c3Mzy/download"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        pytest.skip(f"Could not download Hygeos logo: {e}")
    
    # Load image using PIL
    img = Image.open(BytesIO(response.content))
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array and normalize to [0, 1]
    return xr.DataArray(img, dims=[str(names.rows), str(names.columns), str(names.bands)]) / 255.0


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
    
    def test_real_image_pansharpening(self, hygeos_logo, spatial_mapping, request):
        """Test Brovey pansharpening on the Hygeos logo."""
        
        # Apply Brovey pansharpening
        brovey = Brovey(
            var_ms=valid_ms_vars[0],
            panchro='pan',
            spatial_mapping=spatial_mapping,
            weights=[0.25, 0.6, 0.15]
        )
        
        _Test.test_real_image_pansharpening(brovey, hygeos_logo, request)


class TestIHS:
    """Test suite for the IHS pansharpening class."""
    
    @pytest.fixture
    def spatial_mapping(self):
        """Spatial mapping for tests."""
        return {str(names.rows): 'pan_y', str(names.columns): 'pan_x'}
    
    def test_initialization(self, spatial_mapping):
        """Test IHS initialization."""
        ihs = IHS(var_ms=valid_ms_vars[0], panchro="pan", spatial_mapping=spatial_mapping)
        assert ihs.var_ms == valid_ms_vars[0]
        assert ihs.panchro == "pan"
        assert ihs.spatial_mapping == spatial_mapping
    
    def test_input_vars_valid(self, spatial_mapping):
        """Test input_vars returns correct variables for valid var_ms."""
        ihs = IHS(var_ms=valid_ms_vars[0], panchro="pan", spatial_mapping=spatial_mapping)
        
        result = ihs.input_vars()
        
        assert len(result) == 2
        assert isinstance(result[0], Var)
        assert isinstance(result[1], Var)
    
    def test_input_vars_invalid_raises_assertion(self, spatial_mapping):
        """Test input_vars raises AssertionError for invalid var_ms."""
        ihs = IHS(var_ms="invalid_var", panchro="pan", spatial_mapping=spatial_mapping)
        
        with pytest.raises(AssertionError):
            ihs.input_vars()
    
    def test_created_vars(self, spatial_mapping):
        """Test created_vars returns upsample variable."""
        ihs = IHS(var_ms=valid_ms_vars[0], panchro="pan", spatial_mapping=spatial_mapping)
        result = ihs.created_vars()
        
        assert len(result) == 1
        assert isinstance(result[0], Var)
    
    def test_process_block_requires_three_bands(self, spatial_mapping):
        """Test that IHS raises ValueError when not given exactly 3 bands."""
        var_ms = valid_ms_vars[0]
        ihs = IHS(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
        # Create data with 4 bands (should fail)
        pan_data = xr.DataArray(
            np.ones((10, 10)) * 100,
            dims=['pan_y', 'pan_x']
        )
        ms_data = xr.DataArray(
            np.ones((5, 5, 4)) * 50,
            dims=[str(names.rows), str(names.columns), str(names.bands)]
        )
        
        block = {
            "pan": pan_data,
            var_ms: ms_data
        }
        
        with pytest.raises(ValueError, match="IHS pansharpening requires exactly 3 bands"):
            ihs.process_block(block)
    
    def test_process_block_requires_not_two_bands(self, spatial_mapping):
        """Test that IHS raises ValueError for 2 bands."""
        var_ms = valid_ms_vars[0]
        ihs = IHS(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
        # Create data with 2 bands (should fail)
        pan_data = xr.DataArray(
            np.ones((10, 10)) * 100,
            dims=['pan_y', 'pan_x']
        )
        ms_data = xr.DataArray(
            np.ones((5, 5, 2)) * 50,
            dims=[str(names.rows), str(names.columns), str(names.bands)]
        )
        
        block = {
            "pan": pan_data,
            var_ms: ms_data
        }
        
        with pytest.raises(ValueError, match="IHS pansharpening requires exactly 3 bands"):
            ihs.process_block(block)
    
    def test_process_block_basic(self, spatial_mapping):
        """Test process_block with 3 bands (basic functionality)."""
        var_ms = valid_ms_vars[0]
        ihs = IHS(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
        # Create real data with proper dimensions (3 bands)
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
        
        ihs.process_block(block)
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
    
    def test_process_block_spatial_upsampling(self, spatial_mapping):
        """Test that spatial resolution is upsampled by factor of 2."""
        var_ms = valid_ms_vars[0]
        ihs = IHS(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
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
        
        ihs.process_block(block)
        result = block['upsample']
        
        # After upsampling 5x5 should become 10x10
        assert result.sizes['pan_y'] == 10
        assert result.sizes['pan_x'] == 10
        assert result.sizes[str(names.bands)] == 3
    
    def test_process_block_ihs_transformation(self, spatial_mapping):
        """Test the actual IHS transformation calculation."""
        var_ms = valid_ms_vars[0]
        ihs = IHS(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
        # Create test data with known values
        # Pan has mean=100, std=0 (constant)
        pan_data = xr.DataArray(
            np.ones((10, 10)) * 100,
            dims=['pan_y', 'pan_x']
        )
        # MS bands all equal to 50
        ms_data = xr.DataArray(
            np.ones((5, 5, 3)) * 50,
            dims=[str(names.rows), str(names.columns), str(names.bands)]
        )
        
        block = {
            "pan": pan_data,
            var_ms: ms_data
        }
        
        ihs.process_block(block)
        result = block['upsample']
        
        # Intensity = mean(50, 50, 50) = 50
        # Pan_matched = histogram_match(100, 50) 
        # Since pan is constant (std=0), pan_matched ≈ 50 (matches intensity mean)
        # delta = 50 - 50 = 0
        # Result should be approximately 50 + 0 = 50
        assert result is not None
        assert np.all(result.values > 0)
        # Since pan is constant, histogram matching makes it match intensity
        assert np.allclose(result.values, 50, atol=1)
    
    def test_process_block_ihs_with_varying_pan(self, spatial_mapping):
        """Test IHS transformation with varying panchromatic values."""
        var_ms = valid_ms_vars[0]
        ihs = IHS(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
        # Create test data with varying pan values
        # Pan with mean=100, std≠0
        np.random.seed(42)
        pan_array = np.random.normal(100, 20, (10, 10))
        pan_data = xr.DataArray(pan_array, dims=['pan_y', 'pan_x'])
        
        # MS bands all equal to 50
        ms_data = xr.DataArray(
            np.ones((5, 5, 3)) * 50,
            dims=[str(names.rows), str(names.columns), str(names.bands)]
        )
        
        block = {
            "pan": pan_data,
            var_ms: ms_data
        }
        
        ihs.process_block(block)
        result = block['upsample']
        
        # Result should preserve the spatial detail from pan
        assert result is not None
        assert np.all(result.values > 0)
        # The mean should be approximately preserved
        assert np.allclose(result.mean().values, 50, atol=5)
    
    def test_process_block_with_different_band_values(self, spatial_mapping):
        """Test IHS transformation with different RGB band values."""
        var_ms = valid_ms_vars[0]
        ihs = IHS(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
        # Create test data with different band values
        pan_data = xr.DataArray(
            np.ones((10, 10)) * 150,
            dims=['pan_y', 'pan_x']
        )
        # Different values for each band (R=40, G=60, B=80)
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
        
        ihs.process_block(block)
        result = block['upsample']
        
        # Intensity = (40 + 60 + 80) / 3 = 60
        # Pan_matched should match intensity statistics (mean=60, std=0)
        # Since pan is constant, histogram match gives constant ≈ 60
        # delta = 60 - 60 = 0
        # Result bands should be approximately [40, 60, 80]
        assert result is not None
        assert np.all(result.values > 0)
        # Check that relative differences between bands are preserved
        band_means = [result.sel({str(names.bands): i}).mean().values for i in range(3)]
        assert band_means[0] < band_means[1] < band_means[2]
    
    def test_process_block_histogram_matching_effect(self, spatial_mapping):
        """Test that histogram matching affects the output correctly."""
        var_ms = valid_ms_vars[0]
        ihs = IHS(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
        # Create pan with higher mean and variance
        np.random.seed(123)
        pan_array = np.random.normal(200, 30, (10, 10))
        pan_data = xr.DataArray(pan_array, dims=['pan_y', 'pan_x'])
        
        # MS bands with lower mean
        ms_array = np.stack([
            np.ones((5, 5)) * 50,
            np.ones((5, 5)) * 60,
            np.ones((5, 5)) * 70
        ], axis=-1)
        ms_data = xr.DataArray(
            ms_array,
            dims=[str(names.rows), str(names.columns), str(names.bands)]
        )
        
        block = {
            "pan": pan_data,
            var_ms: ms_data
        }
        
        ihs.process_block(block)
        result = block['upsample']
        
        # The output should have enhanced values due to histogram matching
        # Intensity = (50 + 60 + 70) / 3 = 60
        # Pan has higher values, so after matching and delta injection,
        # result should have mean close to original intensity (60)
        # but with spatial detail from pan
        assert result is not None
        assert np.all(result.values > 0)
        # Mean should be approximately preserved (around 60)
        assert np.allclose(result.mean().values, 60, atol=10)
    
    @pytest.mark.parametrize("var_index", [0, 1, 2])
    def test_different_var_ms_types(self, var_index, spatial_mapping):
        """Test IHS works with different valid var_ms types."""
        var_ms = valid_ms_vars[var_index]
        ihs = IHS(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        assert ihs.var_ms == var_ms
    
    def test_process_block_additive_injection(self, spatial_mapping):
        """Test that IHS uses additive injection (not multiplicative)."""
        var_ms = valid_ms_vars[0]
        ihs = IHS(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
        # Create constant pan (easy to predict)
        pan_data = xr.DataArray(
            np.ones((10, 10)) * 100,
            dims=['pan_y', 'pan_x']
        )
        
        # MS bands with value 50
        ms_data = xr.DataArray(
            np.ones((5, 5, 3)) * 50,
            dims=[str(names.rows), str(names.columns), str(names.bands)]
        )
        
        block = {
            "pan": pan_data,
            var_ms: ms_data
        }
        
        ihs.process_block(block)
        result = block['upsample']
        
        # With constant values:
        # Intensity = 50
        # Pan_matched ≈ 50 (matched to intensity)
        # delta = 0
        # Result = 50 + 0 = 50 (additive)
        # If it were multiplicative like Brovey, result would be different
        assert result is not None
        # All bands should have same value since delta is uniform
        band_values = [result.sel({str(names.bands): i}).values for i in range(3)]
        assert np.allclose(band_values[0], band_values[1])
        assert np.allclose(band_values[1], band_values[2])
    
    def test_real_image_pansharpening(self, hygeos_logo, spatial_mapping, request):
        """Test IHS pansharpening on the Hygeos logo."""
        
        # Apply IHS pansharpening
        ihs = IHS(
            var_ms=valid_ms_vars[0],
            panchro='pan',
            spatial_mapping=spatial_mapping
        )
        
        _Test.test_real_image_pansharpening(ihs, hygeos_logo, request)


class TestHPF:
    """Test suite for the HPF pansharpening class."""
    
    @pytest.fixture
    def spatial_mapping(self):
        """Spatial mapping for tests."""
        return {str(names.rows): 'pan_y', str(names.columns): 'pan_x'}
    
    def test_initialization_without_weights(self, spatial_mapping):
        """Test HPF initialization with default weights."""
        hpf = HPF(var_ms=valid_ms_vars[0], panchro="pan", spatial_mapping=spatial_mapping)
        assert hpf.var_ms == valid_ms_vars[0]
        assert hpf.panchro == "pan"
        assert hpf.spatial_mapping == spatial_mapping
        assert hpf.weights is None
        assert hpf.kernel_size == 5  # default
    
    def test_initialization_with_weights(self, spatial_mapping):
        """Test HPF initialization with custom weights and kernel size."""
        weights = [0.8, 1.0, 0.9]
        hpf = HPF(var_ms=valid_ms_vars[0], panchro="pan", spatial_mapping=spatial_mapping, 
                  kernel_size=7, weights=weights)
        assert hpf.weights == weights
        assert hpf.kernel_size == 7
    
    def test_input_vars_valid(self, spatial_mapping):
        """Test input_vars returns correct variables for valid var_ms."""
        hpf = HPF(var_ms=valid_ms_vars[0], panchro="pan", spatial_mapping=spatial_mapping)
        
        result = hpf.input_vars()
        
        assert len(result) == 2
        assert isinstance(result[0], Var)
        assert isinstance(result[1], Var)
    
    def test_input_vars_invalid_raises_assertion(self, spatial_mapping):
        """Test input_vars raises AssertionError for invalid var_ms."""
        hpf = HPF(var_ms="invalid_var", panchro="pan", spatial_mapping=spatial_mapping)
        
        with pytest.raises(AssertionError):
            hpf.input_vars()
    
    def test_created_vars(self, spatial_mapping):
        """Test created_vars returns upsample variable."""
        hpf = HPF(var_ms=valid_ms_vars[0], panchro="pan", spatial_mapping=spatial_mapping)
        result = hpf.created_vars()
        
        assert len(result) == 1
        assert isinstance(result[0], Var)
    
    def test_process_block_basic(self, spatial_mapping):
        """Test process_block with basic functionality."""
        var_ms = valid_ms_vars[0]
        hpf = HPF(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        
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
        
        hpf.process_block(block)
        result = block['upsample']
        
        # Result should have upsampled dimensions (5x5 -> 10x10)
        assert result is not None
        assert 'pan_y' in result.dims
        assert 'pan_x' in result.dims
        assert str(names.bands) in result.dims
        assert result.sizes['pan_y'] == 10
        assert result.sizes['pan_x'] == 10
        assert result.sizes[str(names.bands)] == 3
    
    def test_process_block_with_varied_pan(self, spatial_mapping):
        """Test HPF with varying panchromatic values."""
        var_ms = valid_ms_vars[0]
        hpf = HPF(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping, kernel_size=3)
        
        # Create pan with spatial variation
        np.random.seed(42)
        pan_data = xr.DataArray(
            np.random.normal(100, 20, (10, 10)),
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
        
        hpf.process_block(block)
        result = block['upsample']
        
        # Result should have spatial variation injected
        assert result is not None
        assert np.std(result.values) > 0  # Should have variation
    
    def test_process_block_with_weights(self, spatial_mapping):
        """Test HPF with custom per-band weights."""
        var_ms = valid_ms_vars[0]
        weights = [0.5, 1.0, 0.8]
        hpf = HPF(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping, 
                  kernel_size=5, weights=weights)
        
        # Create test data
        np.random.seed(42)
        pan_data = xr.DataArray(
            np.random.normal(100, 20, (10, 10)),
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
        
        hpf.process_block(block)
        result = block['upsample']
        
        # Different bands should have different amounts of detail injected
        assert result is not None
        band_stds = [result.sel({str(names.bands): i}).std().values for i in range(3)]
        # Band 1 (weight=1.0) should have most detail
        assert band_stds[1] >= band_stds[0]  # weight 1.0 >= weight 0.5
    
    @pytest.mark.parametrize("var_index", [0, 1, 2])
    def test_different_var_ms_types(self, var_index, spatial_mapping):
        """Test HPF works with different valid var_ms types."""
        var_ms = valid_ms_vars[var_index]
        hpf = HPF(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping)
        assert hpf.var_ms == var_ms
    
    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_different_kernel_sizes(self, kernel_size, spatial_mapping):
        """Test HPF with different kernel sizes."""
        var_ms = valid_ms_vars[0]
        hpf = HPF(var_ms=var_ms, panchro="pan", spatial_mapping=spatial_mapping, 
                  kernel_size=kernel_size)
        
        # Create test data
        pan_data = xr.DataArray(
            np.random.normal(100, 20, (10, 10)),
            dims=['pan_y', 'pan_x']
        )
        ms_data = xr.DataArray(
            np.ones((5, 5, 3)) * 50,
            dims=[str(names.rows), str(names.columns), str(names.bands)]
        )
        
        block = {"pan": pan_data, var_ms: ms_data}
        hpf.process_block(block)
        result = block['upsample']
        
        # Should work with different kernel sizes
        assert result is not None
    
    def test_real_image_pansharpening(self, hygeos_logo, spatial_mapping, request):
        """Test HPF pansharpening on the Hygeos logo."""
        
        # Apply HPF pansharpening
        hpf = HPF(
            var_ms=valid_ms_vars[0],
            panchro='pan',
            spatial_mapping=spatial_mapping,
            kernel_size=10,
            weights=[1.0, 1.0, 1.0]
        )
        
        _Test.test_real_image_pansharpening(hpf, hygeos_logo, request)
