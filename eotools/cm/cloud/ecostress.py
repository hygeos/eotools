from eotools.cm.cloud.utils import stdNxN
from xarray import DataArray


def EcostressV1(
    dem: DataArray, 
    tir1: DataArray,
    tir2: DataArray,
    tir3: DataArray
) -> DataArray:
    """ECOSTRESS Version 1 cloud mask algorithm.
    
    Detects clouds using thermal infrared bands with elevation correction.
    The algorithm uses temperature thresholds and brightness temperature 
    differences between thermal bands to identify cloudy pixels.
    
    Parameters
    ----------
    dem : DataArray
        Digital Elevation Model in kilometers (elevation/1000)
    tir1 : DataArray
        Brightness Temperature of thermal infrared band at 8.7 µm in Kelvin
    tir2 : DataArray
        Brightness Temperature of thermal infrared band at 10.8 µm in Kelvin
    tir3 : DataArray
        Brightness Temperature of thermal infrared band at 12 µm in Kelvin
    
    Returns
    -------
    DataArray
        Cloud mask where True indicates cloud and False indicates clear sky
    """
    
    # Apply thermal cloud detection tests
    test1 = tir2 > 300 - 6*dem          # Elevation-corrected temperature test (K)
    test2 = tir2 - tir3 < 1             # TIR1-TIR2 difference test
    test3 = tir1 - tir2 < -1            # NIR-TIR1 difference test
    
    # Combine tests - any positive test indicates clear sky
    return ~(test1 | test2 | test3)     # Invert to get cloud mask

def Strabala(
    rad8: DataArray,
    tir1: DataArray,
    tir2: DataArray, 
    tir3: DataArray
) -> DataArray:
    """Strabala cloud mask algorithm for thermal data.
    
    Implements the Strabala cloud detection method using thermal band
    statistics and brightness temperature differences. This algorithm
    identifies clouds based on spatial homogeneity and thermal properties.
    
    Parameters
    ----------
    rad8 : DataArray
        Radiance values for thermal infrared band at 8.7 µm
    tir1 : DataArray
        Brightness Temperature of thermal infrared band at 8.7 µm in Kelvin
    tir2 : DataArray
        Brightness Temperature of thermal infrared band at 10.8 µm in Kelvin
    tir3 : DataArray
        Brightness Temperature of thermal infrared band at 12 µm in Kelvin
    
    Returns
    -------
    DataArray
        Cloud mask where True indicates cloud and False indicates clear sky
    """

    # Spatial homogeneity test - low variance indicates uniform (clear) areas
    test1 = stdNxN(rad8, 10) < 0.5
    
    # Thermal band difference tests - small differences indicate clear sky
    test2 = (tir1 - tir2) < 0.5  # TIR1-TIR2 difference
    test3 = (tir2 - tir3) < 2.4  # TIR2-TIR3 difference
    
    # Temperature threshold - warm enough for clear conditions
    test4 = tir2 > 277  # Temperature in Kelvin

    # All tests must pass for clear sky classification
    return ~(test1 & test2 & test3 & test4)