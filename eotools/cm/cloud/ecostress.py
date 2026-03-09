from eotools.cm.cloud.utils import stdNxN

from dataclasses import dataclass, field
from typing import Any

from core.process.blockwise import BlockProcessor
from core.geo.naming import names


@dataclass
class EcostressV1(BlockProcessor):
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
    
    tir1: Any
    tir2: Any
    tir3: Any
    dem: str = field(default='dem', init=False)
        
    def input_vars(self):
        return [self.dem, names.bt]
    
    def modified_vars(self):
        return [names.flags]
    
    def process_block(self, block):
        
        dem = block[self.dem]
        tir1 = block[str(names.bt)].sel({str(names.bands): self.tir1})
        tir2 = block[str(names.bt)].sel({str(names.bands): self.tir2})
        tir3 = block[str(names.bt)].sel({str(names.bands): self.tir3})
    
        # Apply thermal cloud detection tests
        test1 = tir2 > 300 - 6*dem          # Elevation-corrected temperature test (K)
        test2 = tir2 - tir3 < 1             # TIR1-TIR2 difference test
        test3 = tir1 - tir2 < -1            # NIR-TIR1 difference test
        
        # Combine tests - any positive test indicates clear sky
        mask = ~(test1 | test2 | test3)     # Invert to get cloud mask
        self.raiseflag(block, str(names.flags), 'cloud', mask)


@dataclass
class Strabala(BlockProcessor):
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
    
    rad8: Any
    tir1: Any
    tir2: Any
    tir3: Any
        
    def input_vars(self):
        return [names.ltoa, names.bt]
    
    def modified_vars(self):
        return [names.flags]
    
    def process_block(self, block):
        
        rad8 = block[str(names.ltoa)].sel({str(names.bands): self.rad8})
        tir1 = block[str(names.bt)].sel({str(names.bands): self.tir1})
        tir2 = block[str(names.bt)].sel({str(names.bands): self.tir2})
        tir3 = block[str(names.bt)].sel({str(names.bands): self.tir3})
        
        # Spatial homogeneity test - low variance indicates uniform (clear) areas
        test1 = stdNxN(rad8, 10) < 0.5
        
        # Thermal band difference tests - small differences indicate clear sky
        test2 = (tir1 - tir2) < 0.5  # TIR1-TIR2 difference
        test3 = (tir2 - tir3) < 2.4  # TIR2-TIR3 difference
        
        # Temperature threshold - warm enough for clear conditions
        test4 = tir2 > 277  # Temperature in Kelvin

        # All tests must pass for clear sky classification
        mask = ~(test1 & test2 & test3 & test4)
        self.raiseflag(block, str(names.flags), 'cloud', mask)