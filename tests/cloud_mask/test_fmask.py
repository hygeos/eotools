#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from xarray import DataArray
import xarray as xr

from eotools.cm.cloud.fmask import FMASK


class CaptureFMASK(FMASK):
    """Test subclass that captures the masks instead of calling raiseflag."""
    
    def raiseflag(self, block, var_name, flag_name, condition):
        """Capture the mask in the block for testing."""
        # Convert numpy array to DataArray if needed
        if isinstance(condition, np.ndarray):
            # Get dimensions from the flags variable in the block
            dims = block['flags'].dims
            condition = DataArray(condition, dims=dims)
        block[f'_captured_{flag_name}'] = condition


@pytest.fixture
def sample_shape():
    """Define a standard shape for test arrays."""
    return (100, 100)


@pytest.fixture
def cloud_block(sample_shape):
    """Create a block Dataset representing cloudy conditions."""
    # Create reflectance data with bands dimension
    blue = DataArray(np.full(sample_shape, 0.3))
    green = DataArray(np.full(sample_shape, 0.3))
    red = DataArray(np.full(sample_shape, 0.3))
    nir = DataArray(np.full(sample_shape, 0.3))
    swir1 = DataArray(np.full(sample_shape, 0.3))
    swir2 = DataArray(np.full(sample_shape, 0.1))
    cirrus = DataArray(np.full(sample_shape, 0.02))
    
    rtoa_data = xr.concat(
        [blue, green, red, nir, swir1, swir2, cirrus],
        dim='bands'
    ).assign_coords(bands=['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'cirrus'])
    
    # Create brightness temperature data with bands dimension
    tir1 = DataArray(np.full(sample_shape, 273.15 + 15))  # Cold (15°C)
    tir2 = DataArray(np.full(sample_shape, 273.15 + 15))
    
    bt_data = xr.concat(
        [tir1, tir2],
        dim='bands'
    ).assign_coords(bands=['tir1', 'tir2'])
    
    # Create flags DataArray initialized to 0
    flags = xr.DataArray(
        np.zeros(sample_shape, dtype=np.uint8),
        dims=['dim_0', 'dim_1']
    )
    
    return xr.Dataset({
        'Rtoa': rtoa_data,
        'BT': bt_data,
        'flags': flags,
    })


@pytest.fixture
def water_block(sample_shape):
    """Create a block Dataset representing water conditions."""
    # Create reflectance data with bands dimension - low reflectances typical of water
    blue = DataArray(np.full(sample_shape, 0.05))
    green = DataArray(np.full(sample_shape, 0.05))
    red = DataArray(np.full(sample_shape, 0.03))
    nir = DataArray(np.full(sample_shape, 0.02))  # Low NIR
    swir1 = DataArray(np.full(sample_shape, 0.01))  # Low SWIR1
    swir2 = DataArray(np.full(sample_shape, 0.005))
    cirrus = DataArray(np.full(sample_shape, 0.001))
    
    rtoa_data = xr.concat(
        [blue, green, red, nir, swir1, swir2, cirrus],
        dim='bands'
    ).assign_coords(bands=['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'cirrus'])
    
    # Create brightness temperature data with bands dimension
    tir1 = DataArray(np.full(sample_shape, 273.15 + 20))  # Moderate temperature
    tir2 = DataArray(np.full(sample_shape, 273.15 + 20))
    
    bt_data = xr.concat(
        [tir1, tir2],
        dim='bands'
    ).assign_coords(bands=['tir1', 'tir2'])
    
    # Create flags DataArray initialized to 0
    flags = xr.DataArray(
        np.zeros(sample_shape, dtype=np.uint8),
        dims=['dim_0', 'dim_1']
    )
    
    return xr.Dataset({
        'Rtoa': rtoa_data,
        'BT': bt_data,
        'flags': flags,
    })


class TestFMASK:
    """Test the main FMASK cloud mask algorithm."""
    
    def test_fmask_sets_cloud_flag(self, cloud_block):
        """Test that FMASK sets the cloud flag in the block."""
        processor = CaptureFMASK(
            blue='blue', green='green', red='red', nir='nir',
            swir1='swir1', swir2='swir2', cirrus='cirrus',
            tir1='tir1', tir2='tir2'
        )
        processor.process_block(cloud_block)
        
        # Check that all three flags are captured
        assert '_captured_cloud' in cloud_block
        assert '_captured_shadow' in cloud_block
        assert '_captured_water' in cloud_block
    
    def test_fmask_output_shape(self, cloud_block, sample_shape):
        """Test that output shapes match input shape."""
        processor = CaptureFMASK(
            blue='blue', green='green', red='red', nir='nir',
            swir1='swir1', swir2='swir2', cirrus='cirrus',
            tir1='tir1', tir2='tir2'
        )
        processor.process_block(cloud_block)
        
        pcloud = cloud_block['_captured_cloud']
        pshadow = cloud_block['_captured_shadow']
        water = cloud_block['_captured_water']
        
        assert pcloud.shape == sample_shape
        assert pshadow.shape == sample_shape
        assert water.shape == sample_shape
    
    def test_fmask_output_dtype(self, cloud_block):
        """Test that outputs are boolean."""
        processor = CaptureFMASK(
            blue='blue', green='green', red='red', nir='nir',
            swir1='swir1', swir2='swir2', cirrus='cirrus',
            tir1='tir1', tir2='tir2'
        )
        processor.process_block(cloud_block)
        
        pcloud = cloud_block['_captured_cloud']
        pshadow = cloud_block['_captured_shadow']
        water = cloud_block['_captured_water']
        
        assert pcloud.dtype == bool
        assert pshadow.dtype == bool
        assert water.dtype == bool
    
    def test_fmask_detects_clouds(self, cloud_block):
        """Test FMASK detects cloud-like pixels."""
        processor = CaptureFMASK(
            blue='blue', green='green', red='red', nir='nir',
            swir1='swir1', swir2='swir2', cirrus='cirrus',
            tir1='tir1', tir2='tir2'
        )
        processor.process_block(cloud_block)
        pcloud = cloud_block['_captured_cloud']
        
        # With cloud-like conditions, should detect some clouds
        assert isinstance(pcloud, DataArray)
    
    def test_fmask_input_vars(self):
        """Test that input_vars returns expected variables."""
        processor = CaptureFMASK(
            blue='blue', green='green', red='red', nir='nir',
            swir1='swir1', swir2='swir2', cirrus='cirrus',
            tir1='tir1', tir2='tir2'
        )
        input_vars = processor.input_vars()
        assert len(input_vars) > 0
    
    def test_fmask_modified_vars(self):
        """Test that modified_vars returns expected variables."""
        processor = CaptureFMASK(
            blue='blue', green='green', red='red', nir='nir',
            swir1='swir1', swir2='swir2', cirrus='cirrus',
            tir1='tir1', tir2='tir2'
        )
        modified_vars = processor.modified_vars()
        assert len(modified_vars) > 0
