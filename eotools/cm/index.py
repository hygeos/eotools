from xarray import DataArray
    
def NDSI(G: DataArray, SWIR: DataArray) -> DataArray:
    """
    Calculate Normalized Difference Snow Index (NDSI).
    
    NDSI is used to distinguish snow/ice from clouds and other bright surfaces.
    Snow typically has high reflectance in visible bands but low reflectance 
    in shortwave infrared bands.
    
    Args:
        G (xr.DataArray): Green band reflectance (typically ~560nm)
        SWIR (xr.DataArray): Shortwave infrared band reflectance (typically ~1600nm)
        
    Returns:
        xr.DataArray: NDSI values ranging from -1 to 1, where:
            - High values (>0.4) typically indicate snow/ice
            - Low values (<0) typically indicate non-snow surfaces
            
    Formula:
        NDSI = (Green - SWIR) / (Green + SWIR)
    """
    return (G-SWIR)/(G+SWIR)

def NDVI(NIR: DataArray, R: DataArray) -> DataArray:
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    
    NDVI is used to identify and assess vegetation health. Vegetation typically
    has high near-infrared reflectance and low red reflectance due to chlorophyll
    absorption.
    
    Args:
        NIR (xr.DataArray): Near-infrared band reflectance (typically ~865nm)
        R (xr.DataArray): Red band reflectance (typically ~655nm)
        
    Returns:
        xr.DataArray: NDVI values ranging from -1 to 1, where:
            - High values (>0.3) typically indicate healthy vegetation
            - Low values (<0.1) typically indicate non-vegetated surfaces
            - Negative values often indicate water or clouds
            
    Formula:
        NDVI = (NIR - Red) / (NIR + Red)
    """
    return (NIR-R)/(NIR+R)