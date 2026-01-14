#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from xarray import DataArray

from eotools.cm.cloud.ecostress import EcostressV1, Strabala


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
def cloud_data_ecostress(cloud_data):
    """Create DataArrays representing cloudy conditions for Strabala"""
    inputs = ['dem','tir1','tir2','tir3']
    return {i: cloud_data[i] for i in inputs}


@pytest.fixture
def clear_data_ecostress(clear_data):
    """Create DataArrays representing clear sky conditions for Strabala"""
    inputs = ['dem','tir1','tir2','tir3']
    return {i: clear_data[i] for i in inputs}

@pytest.fixture
def cloud_data_strabala(cloud_data):
    """Create DataArrays representing cloudy conditions for Strabala"""
    inputs = ['rad8','tir1','tir2','tir3']
    return {i: cloud_data[i] for i in inputs}


@pytest.fixture
def clear_data_strabala(clear_data):
    """Create DataArrays representing clear sky conditions for Strabala"""
    inputs = ['rad8','tir1','tir2','tir3']
    return {i: clear_data[i] for i in inputs}


class TestEcostressV1:
    """Test suite for EcostressV1 cloud mask algorithm"""
    
    def test_ecostress_returns_dataarray(self, cloud_data_ecostress):
        """Test that EcostressV1 returns a DataArray"""
        result = EcostressV1(**cloud_data_ecostress)
        assert isinstance(result, DataArray)
    
    def test_ecostress_output_shape(self, cloud_data_ecostress):
        """Test that output shape matches input shape"""
        result = EcostressV1(**cloud_data_ecostress)
        expected_shape = cloud_data_ecostress['dem'].shape
        assert result.shape == expected_shape
    
    def test_ecostress_output_dtype(self, cloud_data_ecostress):
        """Test that output is boolean"""
        result = EcostressV1(**cloud_data_ecostress)
        assert result.dtype == bool
    
    def test_ecostress_cold_clouds(self, cloud_data_ecostress):
        """Test detection of cold clouds"""
        result = EcostressV1(**cloud_data_ecostress)
        # All tests False -> inverted to True (clouds detected)
        assert result.all(), "Very cold temperatures should indicate clouds"
    
    def test_ecostress_clear_detection(self, clear_data_ecostress):
        """Test that EcostressV1 detects clouds in cloudy conditions"""
        result = EcostressV1(**clear_data_ecostress)
        # Should detect clouds (at least some True values)
        assert (~result).all(), "EcostressV1 should detect clear in clear sky conditions"


class TestStrabala:
    """Test suite for Strabala cloud mask algorithm"""
    
    def test_strabala_returns_dataarray(self, cloud_data_strabala):
        """Test that Strabala returns a DataArray"""
        result = Strabala(**cloud_data_strabala)
        assert isinstance(result, DataArray)
    
    def test_strabala_output_shape(self, cloud_data_strabala):
        """Test that output shape matches input shape"""
        result = Strabala(**cloud_data_strabala)
        expected_shape = cloud_data_strabala['rad8'].shape
        assert result.shape == expected_shape
    
    def test_strabala_output_dtype(self, cloud_data_strabala):
        """Test that output is boolean"""
        result = Strabala(**cloud_data_strabala)
        assert result.dtype == bool
    
    def test_strabala_cloud_detection(self, cloud_data_strabala):
        """Test that Strabala detects clouds in cloudy conditions"""
        result = Strabala(**cloud_data_strabala)
        # All tests False -> inverted to True (clouds detected)
        assert result.all(), "Very cold temperatures should indicate clouds"
    
    def test_strabala_clear_detection(self, clear_data_strabala):
        """Test that Strabala detects clouds in cloudy conditions"""
        result = Strabala(**clear_data_strabala)
        # Should detect clouds (at least some True values)
        assert (~result).all(), "Strabala should detect clear in clear sky conditions"