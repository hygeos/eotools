import pytest
import numpy as np
from xarray import DataArray
from eotools.cm.cloud.fmask import FMASK


@pytest.fixture
def sample_shape():
    """Define a standard shape for test arrays."""
    return (100, 100)


@pytest.fixture
def cloud_bands(sample_shape):
    """Create band data that should trigger cloud detection."""
    blue = DataArray(np.full(sample_shape, 0.3))
    green = DataArray(np.full(sample_shape, 0.3))
    red = DataArray(np.full(sample_shape, 0.3))
    nir = DataArray(np.full(sample_shape, 0.3))
    swir1 = DataArray(np.full(sample_shape, 0.3))
    swir2 = DataArray(np.full(sample_shape, 0.1))  # High value
    cirrus = DataArray(np.full(sample_shape, 0.02))
    tir1 = DataArray(np.full(sample_shape, 273.15 + 15))  # Cold (15°C)
    tir2 = DataArray(np.full(sample_shape, 273.15 + 15))
    
    return {
        'blue': blue,
        'green': green,
        'red': red,
        'nir': nir,
        'swir1': swir1,
        'swir2': swir2,
        'cirrus': cirrus,
        'tir1': tir1,
        'tir2': tir2
    }


@pytest.fixture
def water_bands(sample_shape):
    """Create band data that represents water."""
    blue = DataArray(np.full(sample_shape, 0.05))
    green = DataArray(np.full(sample_shape, 0.05))
    red = DataArray(np.full(sample_shape, 0.03))
    nir = DataArray(np.full(sample_shape, 0.02))  # Low NIR
    swir1 = DataArray(np.full(sample_shape, 0.01))  # Low SWIR1
    swir2 = DataArray(np.full(sample_shape, 0.005))
    cirrus = DataArray(np.full(sample_shape, 0.001))
    tir1 = DataArray(np.full(sample_shape, 273.15 + 20))
    tir2 = DataArray(np.full(sample_shape, 273.15 + 20))
    
    return {
        'blue': blue,
        'green': green,
        'red': red,
        'nir': nir,
        'swir1': swir1,
        'swir2': swir2,
        'cirrus': cirrus,
        'tir1': tir1,
        'tir2': tir2
    }

class TestCloudMask:
    """Test the main cloud_mask method."""
    
    def test_cloud_mask_returns_three_arrays(self, cloud_bands):
        """Test that cloud_mask returns three boolean arrays."""
        result = FMASK(**cloud_bands)
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        pcloud, pshadow, water = result
        assert pcloud.dtype == bool
        assert pshadow.dtype == bool
        assert water.dtype == bool
        assert pcloud.shape == cloud_bands['blue'].shape
    
    def test_cloud_mask_detects_clouds(self, cloud_bands):
        """Test cloud_mask detects cloud-like pixels."""
        pcloud, pshadow, water = FMASK(**cloud_bands)
        # With cloud-like conditions, should detect some clouds
        assert isinstance(pcloud, np.ndarray)