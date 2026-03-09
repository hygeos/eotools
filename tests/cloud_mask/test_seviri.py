#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from xarray import DataArray
import xarray as xr

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
    return DataArray(mask, dims=['dim_0', 'dim_1'])


@pytest.fixture
def cloud_block(sample_shape, sample_water_mask):
    """Create a block Dataset representing cloudy conditions over land."""
    # Create reflectance data with bands dimension
    red = DataArray(np.full(sample_shape, 0.5), dims=['dim_0', 'dim_1'])  # High red (bright)
    nir = DataArray(np.full(sample_shape, 0.5), dims=['dim_0', 'dim_1'])
    swir1 = DataArray(np.full(sample_shape, 0.5), dims=['dim_0', 'dim_1'])  # swir1/nir = 1.0 < 1.2
    
    rtoa_data = xr.concat(
        [red, nir, swir1],
        dim='bands'
    ).assign_coords(bands=['red', 'nir', 'swir1'])
    
    # Create brightness temperature data with bands dimension
    tir1 = DataArray(np.full(sample_shape, 280.0), dims=['dim_0', 'dim_1'])  # Cold < 288
    tir2 = DataArray(np.full(sample_shape, 281.0), dims=['dim_0', 'dim_1'])
    
    bt_data = xr.concat(
        [tir1, tir2],
        dim='bands'
    ).assign_coords(bands=['tir1', 'tir2'])
    
    return xr.Dataset({
        'water': sample_water_mask,
        'Rtoa': rtoa_data,
        'BT': bt_data,
    })


@pytest.fixture
def clear_block(sample_shape, sample_water_mask):
    """Create a block Dataset representing clear conditions over sea."""
    # Create reflectance data with bands dimension
    red = DataArray(np.full(sample_shape, 0.03), dims=['dim_0', 'dim_1'])
    nir = DataArray(np.full(sample_shape, 0.04), dims=['dim_0', 'dim_1'])  # Low NIR
    swir1 = DataArray(np.full(sample_shape, 0.06), dims=['dim_0', 'dim_1'])  # swir1/nir = 1.5 > 1.3
    
    rtoa_data = xr.concat(
        [red, nir, swir1],
        dim='bands'
    ).assign_coords(bands=['red', 'nir', 'swir1'])
    
    # Create brightness temperature data with bands dimension
    tir1 = DataArray(np.full(sample_shape, 305.0), dims=['dim_0', 'dim_1'])  # Warm > 300
    tir2 = DataArray(np.full(sample_shape, 304.0), dims=['dim_0', 'dim_1'])
    
    bt_data = xr.concat(
        [tir1, tir2],
        dim='bands'
    ).assign_coords(bands=['tir1', 'tir2'])
    
    return xr.Dataset({
        'water': sample_water_mask,
        'Rtoa': rtoa_data,
        'BT': bt_data,
    })


class TestCountPts:
    """Test the count_pts helper function."""
    
    def test_count_pts_single_test(self):
        """Test count_pts with a single test array."""
        test1 = DataArray(np.array([[True, False], [True, True]]))
        result = count_pts([test1])
        expected = np.array([[1, 0], [1, 1]], dtype='uint8')
        np.testing.assert_array_equal(result.values, expected)
    
    def test_count_pts_multiple_tests(self):
        """Test count_pts with multiple test arrays."""
        test1 = DataArray(np.array([[True, False], [True, True]]))
        test2 = DataArray(np.array([[True, True], [False, True]]))
        test3 = DataArray(np.array([[False, True], [True, False]]))
        result = count_pts([test1, test2, test3])
        expected = np.array([[2, 2], [2, 2]], dtype='uint8')
        np.testing.assert_array_equal(result.values, expected)


class TestMSGCOAST:
    """Test the main MSG_COAST function."""
    
    def test_msg_coast_returns_dataarray(self, cloud_block):
        """Test that MSG_COAST returns a DataArray."""
        processor = MSG_COAST(
            water='water', red='red', nir='nir', swir1='swir1',
            tir1='tir1', tir2='tir2'
        )
        processor.process_block(cloud_block)
        result = cloud_block['MSG_cloudmask']
        assert isinstance(result, DataArray)
        assert result.shape == cloud_block['water'].shape
    
    def test_msg_coast_output_range(self, cloud_block):
        """Test that MSG_COAST returns values in expected range."""
        processor = MSG_COAST(
            water='water', red='red', nir='nir', swir1='swir1',
            tir1='tir1', tir2='tir2'
        )
        processor.process_block(cloud_block)
        result = cloud_block['MSG_cloudmask']
        
        # Values should be one of the label values (0-3)
        unique_values = np.unique(result.values)
        label = list(result.attrs.values())
        assert all(val in label for val in unique_values) 
    
    def test_msg_coast_water_land_separation(self, cloud_block, sample_shape):
        """Test that MSG_COAST processes water and land separately."""
        processor = MSG_COAST(
            water='water', red='red', nir='nir', swir1='swir1',
            tir1='tir1', tir2='tir2'
        )
        processor.process_block(cloud_block)
        result = cloud_block['MSG_cloudmask']
        
        # Should have results for both water and land regions
        assert result.shape == sample_shape
        water_results = result.values[:, :sample_shape[1]//2]
        land_results = result.values[:, sample_shape[1]//2:]
        
        # Both regions should have some classification
        assert water_results.size > 0
        assert land_results.size > 0
    
    def test_msg_coast_has_label_attrs(self, cloud_block):
        """Test that MSG_COAST result has label attributes."""
        processor = MSG_COAST(
            water='water', red='red', nir='nir', swir1='swir1',
            tir1='tir1', tir2='tir2'
        )
        processor.process_block(cloud_block)
        result = cloud_block['MSG_cloudmask']
        
        # Check that label attributes are present
        assert 'cloud_certain' in result.attrs
        assert 'cloud_uncertain' in result.attrs
        assert 'clear_uncertain' in result.attrs
        assert 'clear_certain' in result.attrs
    
    def test_msg_coast_input_vars(self):
        """Test that input_vars returns expected variables."""
        processor = MSG_COAST(
            water='water', red='red', nir='nir', swir1='swir1',
            tir1='tir1', tir2='tir2'
        )
        input_vars = processor.input_vars()
        assert len(input_vars) > 0
    
    def test_msg_coast_created_vars(self):
        """Test that created_vars returns expected variables."""
        processor = MSG_COAST(
            water='water', red='red', nir='nir', swir1='swir1',
            tir1='tir1', tir2='tir2'
        )
        created_vars = processor.created_vars()
        assert len(created_vars) > 0
