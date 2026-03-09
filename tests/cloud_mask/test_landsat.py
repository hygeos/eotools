#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from xarray import DataArray
import xarray as xr

from eotools.cm.cloud.landsat import AFAR, ACCAm


class CaptureAFAR(AFAR):
    """Test subclass that captures the cloud mask instead of calling raiseflag."""
    
    def raiseflag(self, block, var_name, flag_name, condition):
        """Capture the mask in the block for testing."""
        block['_captured_mask'] = condition


class CaptureACCAm(ACCAm):
    """Test subclass that captures the cloud mask instead of calling raiseflag."""
    
    def raiseflag(self, block, var_name, flag_name, condition):
        """Capture the mask in the block for testing."""
        block['_captured_mask'] = condition


@pytest.fixture
def sample_shape():
    """Define a standard shape for test arrays."""
    return (5, 5)


@pytest.fixture
def cloud_block_afar(sample_shape):
    """Create a block Dataset representing cloudy conditions for AFAR."""
    # High cirrus values (clouds)
    coastal = DataArray(np.full(sample_shape, 0.25))  # High coastal
    blue = DataArray(np.full(sample_shape, 0.30))     # High blue
    green = DataArray(np.full(sample_shape, 0.30))    # High green
    red = DataArray(np.full(sample_shape, 0.20))      # Moderate red
    nir = DataArray(np.full(sample_shape, 0.35))      # High NIR
    swir1 = DataArray(np.full(sample_shape, 0.40))    # High SWIR1
    swir2 = DataArray(np.full(sample_shape, 0.10))    # High SWIR2
    cirrus = DataArray(np.full(sample_shape, 0.02))   # High cirrus (> 0.01)
    
    rtoa_data = xr.concat(
        [coastal, blue, green, red, nir, swir1, swir2, cirrus],
        dim='bands'
    ).assign_coords(bands=['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'cirrus'])
    
    # Create flags DataArray initialized to 0
    flags = xr.DataArray(
        np.zeros(sample_shape, dtype=np.uint8),
        dims=['dim_0', 'dim_1']
    )
    
    return xr.Dataset({
        'Rtoa': rtoa_data,
        'flags': flags,
    })


@pytest.fixture
def clear_block_afar(sample_shape):
    """Create a block Dataset representing clear sky conditions for AFAR."""
    # Low reflectance values (clear sky)
    coastal = DataArray(np.full(sample_shape, 0.05))
    blue = DataArray(np.full(sample_shape, 0.05))
    green = DataArray(np.full(sample_shape, 0.08))
    red = DataArray(np.full(sample_shape, 0.06))
    nir = DataArray(np.full(sample_shape, 0.45))      # High NIR (vegetation)
    swir1 = DataArray(np.full(sample_shape, 0.20))
    swir2 = DataArray(np.full(sample_shape, 0.01))    # Low SWIR2
    cirrus = DataArray(np.full(sample_shape, 0.005))  # Low cirrus (< 0.01)
    
    rtoa_data = xr.concat(
        [coastal, blue, green, red, nir, swir1, swir2, cirrus],
        dim='bands'
    ).assign_coords(bands=['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'cirrus'])
    
    # Create flags DataArray initialized to 0
    flags = xr.DataArray(
        np.zeros(sample_shape, dtype=np.uint8),
        dims=['dim_0', 'dim_1']
    )
    
    return xr.Dataset({
        'Rtoa': rtoa_data,
        'flags': flags,
    })


@pytest.fixture
def cloud_block_accam(sample_shape):
    """Create a block Dataset representing cloudy conditions for ACCAm."""
    # High cirrus values (clouds)
    coastal = DataArray(np.full(sample_shape, 0.25))  # High coastal
    green = DataArray(np.full(sample_shape, 0.30))    # High green
    red = DataArray(np.full(sample_shape, 0.20))      # Moderate red
    nir = DataArray(np.full(sample_shape, 0.35))      # High NIR
    swir1 = DataArray(np.full(sample_shape, 0.40))    # High SWIR1
    cirrus = DataArray(np.full(sample_shape, 0.02))   # High cirrus (> 0.01)
    
    rtoa_data = xr.concat(
        [coastal, green, red, nir, swir1, cirrus],
        dim='bands'
    ).assign_coords(bands=['coastal', 'green', 'red', 'nir', 'swir1', 'cirrus'])
    
    # Create brightness temperature data with bands dimension
    tir1 = DataArray(np.full(sample_shape, 285.0))    # Cold temperature (clouds)
    
    bt_data = xr.concat(
        [tir1],
        dim='bands'
    ).assign_coords(bands=['tir1'])
    
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
def clear_block_accam(sample_shape):
    """Create a block Dataset representing clear sky conditions for ACCAm."""
    # Low reflectance values (clear sky)
    coastal = DataArray(np.full(sample_shape, 0.05))
    green = DataArray(np.full(sample_shape, 0.08))
    red = DataArray(np.full(sample_shape, 0.06))
    nir = DataArray(np.full(sample_shape, 0.45))      # High NIR (vegetation)
    swir1 = DataArray(np.full(sample_shape, 0.20))
    cirrus = DataArray(np.full(sample_shape, 0.005))  # Low cirrus (< 0.01)
    
    rtoa_data = xr.concat(
        [coastal, green, red, nir, swir1, cirrus],
        dim='bands'
    ).assign_coords(bands=['coastal', 'green', 'red', 'nir', 'swir1', 'cirrus'])
    
    # Create brightness temperature data with bands dimension
    tir1 = DataArray(np.full(sample_shape, 305.0))    # Warm temperature (clear)
    
    bt_data = xr.concat(
        [tir1],
        dim='bands'
    ).assign_coords(bands=['tir1'])
    
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


class TestAFAR:
    """Test suite for AFAR cloud mask algorithm"""
    
    def test_afar_returns_dataarray(self, cloud_block_afar):
        """Test that AFAR returns a DataArray"""
        processor = CaptureAFAR(
            coastal='coastal', blue='blue', green='green', red='red',
            nir='nir', swir1='swir1', swir2='swir2', cirrus='cirrus'
        )
        processor.process_block(cloud_block_afar)
        result = cloud_block_afar['_captured_mask']
        assert isinstance(result, DataArray)
    
    def test_afar_output_shape(self, cloud_block_afar, sample_shape):
        """Test that output shape matches input shape"""
        processor = CaptureAFAR(
            coastal='coastal', blue='blue', green='green', red='red',
            nir='nir', swir1='swir1', swir2='swir2', cirrus='cirrus'
        )
        processor.process_block(cloud_block_afar)
        result = cloud_block_afar['_captured_mask']
        assert result.shape == sample_shape
    
    def test_afar_output_dtype(self, cloud_block_afar):
        """Test that output is boolean"""
        processor = CaptureAFAR(
            coastal='coastal', blue='blue', green='green', red='red',
            nir='nir', swir1='swir1', swir2='swir2', cirrus='cirrus'
        )
        processor.process_block(cloud_block_afar)
        result = cloud_block_afar['_captured_mask']
        assert result.dtype == bool
    
    def test_afar_cloud_detection(self, cloud_block_afar):
        """Test that AFAR detects clouds in cloudy conditions"""
        processor = CaptureAFAR(
            coastal='coastal', blue='blue', green='green', red='red',
            nir='nir', swir1='swir1', swir2='swir2', cirrus='cirrus'
        )
        processor.process_block(cloud_block_afar)
        result = cloud_block_afar['_captured_mask']
        # Should detect clouds (at least some True values)
        assert result.all(), "AFAR should detect clouds in cloudy conditions"
        
    def test_afar_clear_detection(self, clear_block_afar):
        """Test that AFAR detects clear in clear conditions"""
        processor = CaptureAFAR(
            coastal='coastal', blue='blue', green='green', red='red',
            nir='nir', swir1='swir1', swir2='swir2', cirrus='cirrus'
        )
        processor.process_block(clear_block_afar)
        result = clear_block_afar['_captured_mask']
        # Should detect clear (at least some False values)
        assert (~result).all(), "AFAR should detect clear in clear conditions"


class TestACCAm:
    """Test suite for ACCAm cloud mask algorithm"""
    
    def test_accam_returns_dataarray(self, cloud_block_accam):
        """Test that ACCAm returns a DataArray"""
        processor = CaptureACCAm(
            coastal='coastal', green='green', red='red',
            nir='nir', swir1='swir1', cirrus='cirrus', tir1='tir1'
        )
        processor.process_block(cloud_block_accam)
        result = cloud_block_accam['_captured_mask']
        assert isinstance(result, DataArray)
    
    def test_accam_output_shape(self, cloud_block_accam, sample_shape):
        """Test that output shape matches input shape"""
        processor = CaptureACCAm(
            coastal='coastal', green='green', red='red',
            nir='nir', swir1='swir1', cirrus='cirrus', tir1='tir1'
        )
        processor.process_block(cloud_block_accam)
        result = cloud_block_accam['_captured_mask']
        assert result.shape == sample_shape
    
    def test_accam_output_dtype(self, cloud_block_accam):
        """Test that output is boolean"""
        processor = CaptureACCAm(
            coastal='coastal', green='green', red='red',
            nir='nir', swir1='swir1', cirrus='cirrus', tir1='tir1'
        )
        processor.process_block(cloud_block_accam)
        result = cloud_block_accam['_captured_mask']
        assert result.dtype == bool
    
    def test_accam_cloud_detection(self, cloud_block_accam):
        """Test that ACCAm detects clouds in cloudy conditions"""
        processor = CaptureACCAm(
            coastal='coastal', green='green', red='red',
            nir='nir', swir1='swir1', cirrus='cirrus', tir1='tir1'
        )
        processor.process_block(cloud_block_accam)
        result = cloud_block_accam['_captured_mask']
        # Should detect clouds (at least some True values)
        assert result.all(), "ACCAm should detect clouds in cloudy conditions"
        
    def test_accam_clear_detection(self, clear_block_accam):
        """Test that ACCAm detects clear in clear conditions"""
        processor = CaptureACCAm(
            coastal='coastal', green='green', red='red',
            nir='nir', swir1='swir1', cirrus='cirrus', tir1='tir1'
        )
        processor.process_block(clear_block_accam)
        result = clear_block_accam['_captured_mask']
        # Should detect clear (at least some False values)
        assert (~result).all(), "ACCAm should detect clear in clear conditions"
