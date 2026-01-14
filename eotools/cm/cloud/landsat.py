from eotools.cm.index import NDVI, NDSI
from xarray import DataArray


def AFAR(
    coastal: DataArray,
    blue: DataArray,
    green: DataArray, 
    red: DataArray, 
    nir: DataArray,
    swir1: DataArray, 
    swir2: DataArray, 
    cirrus: DataArray,
) -> DataArray:
    """Automated Fmask-based Algorithm for cloud Recognition (AFAR).
    
    Multi-spectral cloud detection algorithm for Landsat data that combines
    cirrus band detection with spectral tests including whiteness, brightness,
    and various band ratio thresholds.
    
    Parameters
    ----------
    coastal : DataArray
        Coastal aerosol band reflectance at 440 nm (0-1)
    blue : DataArray
        Blue band reflectance at 480 nm (0-1)
    green : DataArray
        Green band reflectance at 560 nm (0-1)
    red : DataArray
        Red band reflectance at 655 nm (0-1)
    nir : DataArray
        Near-infrared band reflectance at 865 nm (0-1)
    swir1 : DataArray
        Shortwave infrared band reflectance at 1610 nm (0-1)
    swir2 : DataArray
        Shortwave infrared band reflectance at 2200 nm (0-1)
    cirrus : DataArray
        Cirrus band reflectance at 1375 nm (0-1)
    
    Returns
    -------
    DataArray
        Binary cloud mask where True indicates cloud and False indicates clear
    """
    
    # Compute spectral indices for cloud detection
    band_ndsi = NDSI(green, swir1)  # Normalized Difference Snow Index (separates snow from clouds)
    band_ndvi = NDVI(nir, red)  # Normalized Difference Vegetation Index (excludes vegetation)
    band_ave  = (blue + green + red)/3  # Average visible reflectance for whiteness test
    
    # Apply individual cloud detection tests
    cirrus_cloud = cirrus > 0.01  # Direct cirrus cloud detection using dedicated band
    bright_swir  = swir2 > 0.0215  # Clouds are bright in SWIR
    low_ndsi  = band_ndsi < 0.8  # Exclude snow (high NDSI values)
    low_ndvi  = band_ndvi < 0.8  # Exclude dense vegetation (high NDVI values)
    
    # Whiteness test - clouds have similar reflectance across visible bands
    triple = (abs(blue - band_ave) + abs(green - band_ave) + abs(red - band_ave))
    white = triple/band_ave < 0.7  # Low spectral variability indicates white clouds
    
    # Additional spectral tests for cloud identification
    haze_opt = blue - 0.5*red > 0.08  # Haze and thin clouds scatter more blue light
    div_nir = nir / swir1 > 0.75  # Cloud spectral signature in NIR/SWIR ratio
    bright_coast = coastal > 0.2  # Clouds bright in coastal aerosol band
    min_NDSI = band_ndsi > -0.17  # Minimum NDSI threshold to exclude certain surfaces
    
    # Combine all tests into final cloud mask (excluding cirrus)
    merge_test = bright_swir & low_ndsi & low_ndvi & white & \
                 haze_opt & div_nir & bright_coast & min_NDSI
    
    # Final cloud mask combines cirrus detection with multi-spectral tests
    return cirrus_cloud | merge_test

    

def ACCAm(
    coastal: DataArray,
    blue: DataArray,
    green: DataArray,
    red: DataArray,
    nir: DataArray,
    swir1: DataArray,
    cirrus: DataArray,
    tir1: DataArray
) -> DataArray:
    """Automated Cloud-Cover Assessment modified (ACCAm) algorithm.
    
    Modified version of the ACCA algorithm that uses both spectral and thermal
    tests to identify clouds. Combines cirrus detection with NDSI, temperature,
    and various band ratio tests.
    
    Parameters
    ----------
    coastal : DataArray
        Coastal aerosol band reflectance at 440 nm (0-1)
    blue : DataArray
        Blue band reflectance at 480 nm (0-1)
    green : DataArray
        Green band reflectance at 560 nm (0-1)
    red : DataArray
        Red band reflectance at 655 nm (0-1)
    nir : DataArray
        Near-infrared band reflectance at 865 nm (0-1)
    swir1 : DataArray
        Shortwave infrared band reflectance at 1610 nm (0-1)
    cirrus : DataArray
        Cirrus band reflectance at 1375 nm (0-1)
    tir1 : DataArray
        Brightness temperature of thermal infrared band at 10.8 µm in Kelvin
    
    Returns
    -------
    DataArray
        Binary cloud mask where True indicates cloud and False indicates clear
    """

    # Compute spectral indices
    band_ndsi = NDSI(green, swir1)  # Normalized Difference Snow Index
    
    # Direct cirrus detection using dedicated cirrus band
    cirrus_cloud = cirrus >= 0.025
    
    # Spectral and thermal tests to identify clear (non-cloud) pixels
    test3  = red < 0.72  # Non-bright surfaces in red band (clear)
    test4  = (band_ndsi < -0.25) & (band_ndsi > 0.7)  # NDSI range excludes clouds (identifies snow or certain surfaces)
    test5  = tir1 >= 300.  # Warm surfaces (≥300K) typically clear
    test6  = (1 - swir1) * tir1 < 225.  # Combined SWIR-thermal test for clear conditions
    test7  = swir1 < 0.08  # Low SWIR reflectance (shadows, water, vegetation - clear)
    test8  = nir / red >= 2.35  # High NIR/Red ratio (vegetation - clear)
    test9  = nir / green >= 2.16248  # High NIR/Green ratio (vegetation - clear)
    test10 = nir / swir1 <= 0.7  # Low NIR/SWIR ratio (certain clear surfaces)
    test11 = coastal < 0.22  # Low coastal aerosol (clear atmosphere)
    
    # Combine vegetation tests with thermal test
    test6_bis = (test8 | test9 | test10) & test6  # Vegetation with appropriate temperature

    # Combine all tests that indicate clear pixels
    no_cloud = test6_bis | test4 | test5 | test7 | test11
    no_cloud = ~no_cloud  # Invert: if not clearly identified as clear, consider it cloud
    
    # Final cloud mask combines cirrus detection with spectral/thermal tests
    cm = cirrus_cloud | no_cloud

    return cm
