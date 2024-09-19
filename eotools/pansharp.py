from core.naming import names
import numpy as np


def brovey(level1, weights: list = None, srfs = None):
    """
    Simple implementation of weighted Brovey transformation 

    Args:
        level1 (xr.Dataset): level1 dataset outputed by eoread
        weights (list): weights used to compute the pseudo-pan
    """
    
    assert 'panchromatic' in level1, 'No panchromatic image found in level1'
    panchromatic = level1['panchromatic']
    
    # Collect interesting bands
    if names.rtoa in level1: multi = level1[names.rtoa]
    elif names.bt in level1: multi = level1[names.bt]
    else: raise ValueError
    
    # Upscale multi spectral bands spatial resolution and transpose dimensions
    
    
    # Compute pseudo-panchromatic
    if weights: 
        pseudopan = (multi*weights).sum(axis=2)
        pseudopan = np.true_divide(pseudopan, weights.sum())
    else: pseudopan = multi.sum(axis=2)
    
    # Apply Brovey transformation
    ratio = panchromatic/pseudopan
    pansharp = multi*ratio
    
    return pansharp