#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from xarray import DataArray

from eotools.cm.cloud.landsat import AFAR, ACCAm


@pytest.fixture
def cloud_data():
    shape = (5, 5)
    
    # High cirrus values (clouds)
    coastal = DataArray(np.full(shape, 0.25))  # High coastal
    blue = DataArray(np.full(shape, 0.30))     # High blue
    green = DataArray(np.full(shape, 0.30))    # High green
    red = DataArray(np.full(shape, 0.20))      # Moderate red
    nir = DataArray(np.full(shape, 0.35))      # High NIR
    swir1 = DataArray(np.full(shape, 0.40))    # High SWIR1
    swir2 = DataArray(np.full(shape, 0.10))    # High SWIR2
    cirrus = DataArray(np.full(shape, 0.02))   # High cirrus (> 0.01)
    tir1 = DataArray(np.full(shape, 285.0))    # Cold temperature (clouds)
    
    return {
        'coastal': coastal,
        'blue': blue,
        'green': green,
        'red': red,
        'nir': nir,
        'swir1': swir1,
        'swir2': swir2,
        'cirrus': cirrus,
        'tir1': tir1
    }


@pytest.fixture
def clear_data():
    shape = (5, 5)
    
    # Low reflectance values (clear sky)
    coastal = DataArray(np.full(shape, 0.05))
    blue = DataArray(np.full(shape, 0.05))
    green = DataArray(np.full(shape, 0.08))
    red = DataArray(np.full(shape, 0.06))
    nir = DataArray(np.full(shape, 0.45))      # High NIR (vegetation)
    swir1 = DataArray(np.full(shape, 0.20))
    swir2 = DataArray(np.full(shape, 0.01))    # Low SWIR2
    cirrus = DataArray(np.full(shape, 0.005))  # Low cirrus (< 0.01)
    tir1 = DataArray(np.full(shape, 285.0))    # Cold temperature (clouds)
    
    return {
        'coastal': coastal,
        'blue': blue,
        'green': green,
        'red': red,
        'nir': nir,
        'swir1': swir1,
        'swir2': swir2,
        'cirrus': cirrus,
        'tir1': tir1
    }

@pytest.fixture
def cloud_data_afar(cloud_data):
    """Create DataArrays representing cloudy conditions for AFAR"""
    inputs = ['coastal','blue','green','red','nir','swir1','swir2','cirrus']
    return {i: cloud_data[i] for i in inputs}

@pytest.fixture
def clear_data_afar(clear_data):
    """Create DataArrays representing clear sky conditions for AFAR"""
    inputs = ['coastal','blue','green','red','nir','swir1','swir2','cirrus']
    return {i: clear_data[i] for i in inputs}

@pytest.fixture
def cloud_data_accam(cloud_data):
    """Create DataArrays representing cloudy conditions for ACCAm"""
    inputs = ['coastal','blue','green','red','nir','swir1','cirrus','tir1']
    return {i: cloud_data[i] for i in inputs}

@pytest.fixture
def clear_data_accam(clear_data):
    """Create DataArrays representing clear sky conditions for ACCAm"""
    inputs = ['coastal','blue','green','red','nir','swir1','cirrus','tir1']
    return {i: clear_data[i] for i in inputs}


class TestAFAR:
    """Test suite for AFAR cloud mask algorithm"""
    
    def test_afar_returns_dataarray(self, cloud_data_afar):
        """Test that AFAR returns a DataArray"""
        result = AFAR(**cloud_data_afar)
        assert isinstance(result, DataArray)
    
    def test_afar_output_shape(self, cloud_data_afar):
        """Test that output shape matches input shape"""
        result = AFAR(**cloud_data_afar)
        expected_shape = cloud_data_afar['blue'].shape
        assert result.shape == expected_shape
    
    def test_afar_output_dtype(self, cloud_data_afar):
        """Test that output is boolean"""
        result = AFAR(**cloud_data_afar)
        assert result.dtype == bool
    
    def test_afar_cloud_detection(self, cloud_data_afar):
        """Test that AFAR detects clouds in cloudy conditions"""
        result = AFAR(**cloud_data_afar)
        # Should detect clouds (at least some True values)
        assert result.all(), "AFAR should detect clouds in cloudy conditions"
        
    def test_afar_clear_detection(self, clear_data_afar):
        result = AFAR(**clear_data_afar)
        # Should detect clouds (at least some True values)
        assert (~result).all(), "AFAR should detect clear in clear conditions"


class TestACCAm:
    """Test suite for ACCAm cloud mask algorithm"""
    
    def test_accam_returns_dataarray(self, cloud_data_accam):
        """Test that ACCAm returns a DataArray"""
        result = ACCAm(**cloud_data_accam)
        assert isinstance(result, DataArray)
    
    def test_accam_output_shape(self, cloud_data_accam):
        """Test that output shape matches input shape"""
        result = ACCAm(**cloud_data_accam)
        expected_shape = cloud_data_accam['blue'].shape
        assert result.shape == expected_shape
    
    def test_accam_output_dtype(self, cloud_data_accam):
        """Test that output is boolean"""
        result = ACCAm(**cloud_data_accam)
        assert result.dtype == bool
    
    def test_accam_cloud_detection(self, cloud_data_accam):
        """Test that ACCAm detects clouds in cloudy conditions"""
        result = ACCAm(**cloud_data_accam)
        # Should detect clouds (at least some True values)
        assert result.all(), "ACCAm should detect clouds in cloudy conditions"
        
    def test_accam_clear_detection(self, clear_data_accam):
        result = ACCAm(**clear_data_accam)
        # Should detect clouds (at least some True values)
        assert (~result).all(), "ACCAm should detect clear in clear conditions"