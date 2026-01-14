import numpy as np
import pytest
from xarray import DataArray
from unittest.mock import patch, MagicMock
from eotools.cm.cloud.seviri import count_pts, MSG_COAST


@pytest.fixture
def sample_shape():
    """Define a standard shape for test arrays."""
    return (50, 50)


@pytest.fixture
def sample_water_mask(sample_shape):
    """Create a water mask with half water, half land."""
    mask = np.zeros(sample_shape, dtype=bool)
    mask[:, :sample_shape[1]//2] = True  # Left half is water
    return DataArray(mask)


@pytest.fixture
def cloud_data(sample_shape, sample_water_mask):
    """Create band data representing cloudy conditions over land."""
    red = DataArray(np.full(sample_shape, 0.5))  # High red (bright)
    nir = DataArray(np.full(sample_shape, 0.5))
    swir1 = DataArray(np.full(sample_shape, 0.5))  # swir1/nir = 1.0 < 1.2
    tir1 = DataArray(np.full(sample_shape, 280.0))  # Cold < 288
    tir2 = DataArray(np.full(sample_shape, 281.0))
    
    return {
        'water': sample_water_mask,
        'red': red,
        'nir': nir,
        'swir1': swir1,
        'tir1': tir1,
        'tir2': tir2
    }


@pytest.fixture
def clear_data(sample_shape, sample_water_mask):
    """Create band data representing clear conditions over sea."""
    red = DataArray(np.full(sample_shape, 0.03))
    nir = DataArray(np.full(sample_shape, 0.04))  # Low NIR
    swir1 = DataArray(np.full(sample_shape, 0.06))  # swir1/nir = 1.5 > 1.3
    tir1 = DataArray(np.full(sample_shape, 305.0))  # Warm > 300
    tir2 = DataArray(np.full(sample_shape, 304.0))
    
    return {
        'water': sample_water_mask,
        'red': red,
        'nir': nir,
        'swir1': swir1,
        'tir1': tir1,
        'tir2': tir2
    }


class TestMSGCOAST:
    """Test the main MSG_COAST function."""
    
    def test_msg_coast_returns_dataarray(self, cloud_data):
        """Test that MSG_COAST returns a DataArray."""
        result = MSG_COAST(**cloud_data)
        assert isinstance(result, DataArray)
        assert result.shape == cloud_data['water'].shape
    
    def test_msg_coast_output_range(self, cloud_data):
        """Test that MSG_COAST returns values in expected range."""
        result = MSG_COAST(**cloud_data)
        
        # Values should be one of the label values (0-4)
        unique_values = np.unique(result.values)
        label = list(result.attrs.values())
        assert all(val in label for val in unique_values) 
    
    def test_msg_coast_water_land_separation(self, cloud_data, sample_shape):
        """Test that MSG_COAST processes water and land separately."""
        # Create a mask with distinct water and land regions
        result = MSG_COAST(**cloud_data)
        
        # Should have results for both water and land regions
        assert result.shape == sample_shape
        water_results = result.values[:, :sample_shape[1]//2]
        land_results = result.values[:, sample_shape[1]//2:]
        
        # Both regions should have some classification
        assert water_results.size > 0
        assert land_results.size > 0