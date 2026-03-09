#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from xarray import DataArray
import xarray as xr

from eotools.cm.cloud.ecostress import EcostressV1, Strabala


class CaptureEcostressV1(EcostressV1):
    """Test subclass that captures the cloud mask instead of calling raiseflag."""
    
    def raiseflag(self, block, var_name, flag_name, condition):
        """Capture the mask in the block for testing."""
        block['_captured_mask'] = condition


class CaptureStrabala(Strabala):
    """Test subclass that captures the cloud mask instead of calling raiseflag."""
    
    def raiseflag(self, block, var_name, flag_name, condition):
        """Capture the mask in the block for testing."""
        block['_captured_mask'] = condition


@pytest.fixture
def cloud_data():
    """Create DataArrays representing cloudy conditions for EcostressV1"""
    shape = (5, 5)
    dem = DataArray(np.full(shape, 0.5))        # Low elevation
    rad8 = DataArray(np.random.uniform(0.5, 3.0, shape))  # High variability
    tir1 = DataArray(np.full(shape, 270.0))     # Cold
    tir2 = DataArray(np.full(shape, 270.0))     # Cold
    tir3 = DataArray(np.full(shape, 268.0))     # Cold
    
    return {
        'dem': dem,
        'rad8': rad8,
        'tir1': tir1,
        'tir2': tir2,
        'tir3': tir3
    }


@pytest.fixture
def clear_data():
    """Create DataArrays representing clear sky conditions for EcostressV1"""
    shape = (5, 5)
    dem = DataArray(np.full(shape, 0.0))        # Sea level
    rad8 = DataArray(np.full(shape, 1.0) + np.random.normal(0, 0.1, shape))  # Low variability
    tir1 = DataArray(np.full(shape, 305.0))     # Warm
    tir2 = DataArray(np.full(shape, 305.0))     # Warm
    tir3 = DataArray(np.full(shape, 304.0))     # Slightly cooler
    
    return {
        'dem': dem,
        'rad8': rad8,
        'tir1': tir1,
        'tir2': tir2,
        'tir3': tir3
    }


@pytest.fixture
def cloud_block_ecostress(cloud_data):
    """Create a block Dataset representing cloudy conditions for EcostressV1 BlockProcessor"""
    # Create brightness temperature DataArray with bands dimension
    bt_data = xr.concat(
        [cloud_data['tir1'], cloud_data['tir2'], cloud_data['tir3']],
        dim='bands'
    ).assign_coords(bands=['8.7um', '10.8um', '12um'])
    
    # Create flags DataArray initialized to 0
    flags = xr.DataArray(
        np.zeros(cloud_data['dem'].shape, dtype=np.uint8),
        dims=cloud_data['dem'].dims
    )
    
    return xr.Dataset({
        'dem': cloud_data['dem'],
        'BT': bt_data,
        'flags': flags,
    })


@pytest.fixture
def clear_block_ecostress(clear_data):
    """Create a block Dataset representing clear sky conditions for EcostressV1 BlockProcessor"""
    # Create brightness temperature DataArray with bands dimension
    bt_data = xr.concat(
        [clear_data['tir1'], clear_data['tir2'], clear_data['tir3']],
        dim='bands'
    ).assign_coords(bands=['8.7um', '10.8um', '12um'])
    
    # Create flags DataArray initialized to 0
    flags = xr.DataArray(
        np.zeros(clear_data['dem'].shape, dtype=np.uint8),
        dims=clear_data['dem'].dims
    )
    
    return xr.Dataset({
        'dem': clear_data['dem'],
        'BT': bt_data,
        'flags': flags,
    })


@pytest.fixture
def cloud_block_strabala(cloud_data):
    """Create a block Dataset representing cloudy conditions for Strabala"""
    # Create brightness temperature DataArray with bands dimension
    bt_data = xr.concat(
        [cloud_data['tir1'], cloud_data['tir2'], cloud_data['tir3']],
        dim='bands'
    ).assign_coords(bands=['8.7um', '10.8um', '12um'])
    
    # Create radiance DataArray with bands dimension
    ltoa_data = cloud_data['rad8'].expand_dims(bands=['8.7um'])
    
    return xr.Dataset({
        'BT': bt_data,
        'Ltoa': ltoa_data,
    })


@pytest.fixture
def clear_block_strabala(clear_data):
    """Create a block Dataset representing clear sky conditions for Strabala"""
    # Create brightness temperature DataArray with bands dimension
    bt_data = xr.concat(
        [clear_data['tir1'], clear_data['tir2'], clear_data['tir3']],
        dim='bands'
    ).assign_coords(bands=['8.7um', '10.8um', '12um'])
    
    # Create radiance DataArray with bands dimension
    ltoa_data = clear_data['rad8'].expand_dims(bands=['8.7um'])
    
    return xr.Dataset({
        'BT': bt_data,
        'Ltoa': ltoa_data,
    })


class TestEcostressV1:
    """Test suite for EcostressV1 cloud mask algorithm"""
    
    def test_ecostress_sets_cloud_flag(self, cloud_block_ecostress):
        """Test that EcostressV1 sets the cloud flag in the block"""
        processor = CaptureEcostressV1(tir1='8.7um', tir2='10.8um', tir3='12um')
        processor.process_block(cloud_block_ecostress)
        cloud_mask = cloud_block_ecostress['_captured_mask']
        assert isinstance(cloud_mask, DataArray)
    
    def test_ecostress_output_shape(self, cloud_block_ecostress):
        """Test that output shape matches input shape"""
        processor = CaptureEcostressV1(tir1='8.7um', tir2='10.8um', tir3='12um')
        processor.process_block(cloud_block_ecostress)
        cloud_mask = cloud_block_ecostress['_captured_mask']
        expected_shape = cloud_block_ecostress['dem'].shape
        assert cloud_mask.shape == expected_shape
    
    def test_ecostress_output_dtype(self, cloud_block_ecostress):
        """Test that cloud flag is boolean"""
        processor = CaptureEcostressV1(tir1='8.7um', tir2='10.8um', tir3='12um')
        processor.process_block(cloud_block_ecostress)
        cloud_mask = cloud_block_ecostress['_captured_mask']
        assert cloud_mask.dtype == bool
    
    def test_ecostress_cold_clouds(self, cloud_block_ecostress):
        """Test detection of cold clouds"""
        processor = CaptureEcostressV1(tir1='8.7um', tir2='10.8um', tir3='12um')
        processor.process_block(cloud_block_ecostress)
        cloud_mask = cloud_block_ecostress['_captured_mask']
        # All tests False -> inverted to True (clouds detected)
        assert cloud_mask.all(), "Very cold temperatures should indicate clouds"
    
    def test_ecostress_clear_detection(self, clear_block_ecostress):
        """Test that EcostressV1 detects clear in clear sky conditions"""
        processor = CaptureEcostressV1(tir1='8.7um', tir2='10.8um', tir3='12um')
        processor.process_block(clear_block_ecostress)
        cloud_mask = clear_block_ecostress['_captured_mask']
        # Should detect clear sky (False values in cloud mask)
        assert (~cloud_mask).all(), "EcostressV1 should detect clear in clear sky conditions"
    
    def test_ecostress_input_vars(self):
        """Test that input_vars returns expected variables"""
        processor = CaptureEcostressV1(tir1='8.7um', tir2='10.8um', tir3='12um')
        input_vars = processor.input_vars()
        assert 'dem' in input_vars
    
    def test_ecostress_modified_vars(self):
        """Test that modified_vars returns expected variables"""
        processor = CaptureEcostressV1(tir1='8.7um', tir2='10.8um', tir3='12um')
        modified_vars = processor.modified_vars()
        assert len(modified_vars) > 0


class TestStrabala:
    """Test suite for Strabala cloud mask algorithm"""
    
    def test_strabala_returns_dataarray(self, cloud_block_strabala):
        """Test that Strabala returns a DataArray"""
        processor = CaptureStrabala(rad8='8.7um', tir1='8.7um', tir2='10.8um', tir3='12um')
        processor.process_block(cloud_block_strabala)
        result = cloud_block_strabala['_captured_mask']
        assert isinstance(result, DataArray)
    
    def test_strabala_output_shape(self, cloud_block_strabala):
        """Test that output shape matches input shape"""
        processor = CaptureStrabala(rad8='8.7um', tir1='8.7um', tir2='10.8um', tir3='12um')
        processor.process_block(cloud_block_strabala)
        result = cloud_block_strabala['_captured_mask']
        expected_shape = cloud_block_strabala['BT'].sel(bands='8.7um').shape
        assert result.shape == expected_shape
    
    def test_strabala_output_dtype(self, cloud_block_strabala):
        """Test that output is boolean"""
        processor = CaptureStrabala(rad8='8.7um', tir1='8.7um', tir2='10.8um', tir3='12um')
        processor.process_block(cloud_block_strabala)
        result = cloud_block_strabala['_captured_mask']
        assert result.dtype == bool
    
    def test_strabala_cloud_detection(self, cloud_block_strabala):
        """Test that Strabala detects clouds in cloudy conditions"""
        processor = CaptureStrabala(rad8='8.7um', tir1='8.7um', tir2='10.8um', tir3='12um')
        processor.process_block(cloud_block_strabala)
        result = cloud_block_strabala['_captured_mask']
        # All tests False -> inverted to True (clouds detected)
        assert result.all(), "Very cold temperatures should indicate clouds"
    
    def test_strabala_clear_detection(self, clear_block_strabala):
        """Test that Strabala detects clear in clear sky conditions"""
        processor = CaptureStrabala(rad8='8.7um', tir1='8.7um', tir2='10.8um', tir3='12um')
        processor.process_block(clear_block_strabala)
        result = clear_block_strabala['_captured_mask']
        # Should detect clear sky (False values in cloud mask)
        assert (~result).all(), "Strabala should detect clear in clear sky conditions"
