from eotools.cm.cloud.utils import stdNxN
from xarray import DataArray, where


def count_pts(list_test: list) -> DataArray:
    """
    Count the number of positive tests for each pixel.
    """
    output = list_test[0].astype('uint8')
    for test in list_test[1:]:
            output += test.astype('uint8')
    return output

def MSG_COAST(
    water: DataArray,
    red: DataArray,
    nir: DataArray, 
    swir1: DataArray,
    tir1: DataArray,
    tir2: DataArray,
) -> DataArray:
    """MSG COAST cloud detection algorithm for SEVIRI data.
    
    Comprehensive cloud detection algorithm that applies different spectral tests
    for sea and land surfaces. Returns a confidence-weighted cloud mask with four
    classification levels: cloud certain, cloud uncertain, clear uncertain, and
    clear certain.
    
    Parameters
    ----------
    water : DataArray
        Binary water/land mask where True indicates water and False indicates land
    red : DataArray
        Red band reflectance at 655 nm (0-1)
    nir : DataArray
        Near-infrared band reflectance at 865 nm (0-1)
    swir1 : DataArray
        Shortwave infrared band reflectance at 1610 nm (0-1)
    tir1 : DataArray
        Brightness temperature of thermal infrared band at 10.8 µm in Kelvin
    tir2 : DataArray
        Brightness temperature of thermal infrared band at 12 µm in Kelvin
    
    Returns
    -------
    DataArray
        Cloud confidence mask with values:
        - 1.0: cloud certain
        - 0.75: cloud uncertain
        - 0.25: clear uncertain
        - 0.0: clear certain
        Mask includes these labels in its attributes.
    
    Notes
    -----
    The algorithm uses different sets of spectral tests for sea and land surfaces:
    - Sea tests focus on NIR brightness, texture, and thermal properties
    - Land tests incorporate red band reflectance, texture, and band ratios
    """
    
    # Define confidence levels for cloud classification
    label = dict(
        cloud_certain=1,
        cloud_uncertain=0.75,
        clear_uncertain=0.25,
        clear_certain=0,
    )
    
    ## OVER SEA - Apply sea-specific cloud detection tests
    
    # Compute derived bands for sea tests
    std_865 = stdNxN(nir, 4, None, fillv=0)  # NIR spatial variability (4x4 window)
    sub_11_12 = tir1 - tir2  # Brightness temperature difference between TIR bands

    # Sea cloud detection tests
    test1 = (swir1/nir < 1.2) & (nir > 0.05) & (tir1 < 288)  # Cold, bright NIR, low SWIR/NIR ratio (cloud)
    test2 = (std_865 > 0.005) | (nir > 0.1)  # High texture or very bright NIR (cloud)
    test3 = swir1 > 0.1  # Bright SWIR (cloud)
    test4 = (sub_11_12 < -1.) & (tir1 > 300.)  # Negative BTD with warm temperature (cloud)
    
    # Sea clear detection tests
    test5 = (swir1/nir > 1.3) | (tir1 > 300.)  # High SWIR/NIR ratio or warm (clear)
    test6 = (red/nir < 0.7) | (nir < 0.05)  # Low red/NIR or dark NIR (clear water)
    test7 = (std_865/nir > 0.) & (std_865/nir < 0.01) & (nir < 0.1)  # Low relative texture with dark NIR (clear)
    
    # Create unique test identifiers and count positive tests
    id_cloud = test1 + 10*test2 + 100*test3 + 1000*test4  # Weighted sum for cloud tests
    id_clear = test5 + 10*test6 + 100*test7  # Weighted sum for clear tests
    cpt_cloud = count_pts([test1, test2, test3, test4])  # Count of positive cloud tests
    cpt_clear = count_pts([test5, test6, test7])  # Count of positive clear tests

    # Classify pixels based on test results and confidence
    choice = id_clear < id_cloud  # Choose cloud if cloud evidence stronger
    clear  = (id_cloud == 0) & (cpt_clear > 1)  # No cloud tests passed, multiple clear tests passed
    cloudy = ((id_clear == 0) & (cpt_cloud > 1)) | ((id_clear <= 1) & (cpt_cloud > 2))  # Strong cloud evidence

    # Assign confidence levels based on test results
    cloud_certain   = choice & cloudy  # Cloud choice with strong cloud evidence
    cloudy.values   = ~cloudy.values
    cloud_uncertain = choice & cloudy  # Cloud choice without strong evidence
    choice.values   = ~choice.values
    clear_certain   = choice & clear  # Clear choice with strong clear evidence
    clear.values    = ~clear.values
    clear_uncertain = choice & clear  # Clear choice without strong evidence

    # Combine confidence classifications into single sea mask
    sea_mask = label['cloud_certain']*cloud_certain + \
               label['cloud_uncertain']*cloud_uncertain + \
               label['clear_certain']*clear_certain + \
               label['clear_uncertain']*clear_uncertain
    
    ## OVER LAND - Apply land-specific cloud detection tests
    
    # Compute derived bands for land tests
    std_670 = stdNxN(red, 4, None, fillv=0)  # Red spatial variability (4x4 window)
    sub_11_12 = tir1 - tir2  # Brightness temperature difference between TIR bands
    dummy_ratio = (nir - red) / (nir - swir1)  # Normalized difference-based ratio

    # Land cloud detection tests
    test1 = (swir1 / nir < 1.2) & (red > 0.2) & (tir1 < 288.)  # Bright red, low SWIR/NIR, cold (cloud)
    test2 = red > 0.45  # Very bright red (cloud)
    test3 = (red > 0.35) & (std_670 > 0.01)  # Bright red with high texture (cloud)
    test4 = (red > 0.15) & (red / nir < 1.1) & \
            (red / nir > 1.0) & (tir1 > 295.)  # Moderate brightness, specific red/NIR ratio (cloud)
    test5 = (red > 0.2) & (std_670 > 0.005) & (red / swir1 > 1.4)  # Bright red with texture and high red/SWIR (cloud)
    test6 = (sub_11_12 < -1.) & (tir1 < 300.)  # Negative BTD with cool temperature (cloud)
    
    # Land clear detection tests
    test8 = (swir1 / nir > 1.3) | (tir1 > 300.)  # High SWIR/NIR ratio or warm (clear)
    test9 = red / nir < 0.7  # Low red/NIR ratio (vegetation - clear)
    test10 = (std_670 > 0.) & (std_670 < 0.1) & (red > 0.2) & (red < 0.5) \
             & (swir1/red > 1.5) & (swir1/red < 10.)  # Moderate texture and brightness with specific SWIR/red (clear)
    test11 = swir1 / red > 2.2  # Very high SWIR/red ratio (clear soil/rock)
    test12 = (sub_11_12 > 1.) & (tir1 > 300.)  # Positive BTD with warm temperature (clear)
    test13 = (dummy_ratio < -2.) | (dummy_ratio > 2.)  # Extreme normalized ratio (clear anomaly)
    
    # Create unique test identifiers and count positive tests
    id_cloud = test1 + 10*test2 + 10*test3 + 10*test4 + \
                100*test5 + 1000*test6  # Weighted sum for cloud tests
    id_clear = test8 + 10*test9 + 100*test10 + 100*test11 + \
                1000*test12 + 1000*test13  # Weighted sum for clear tests
    
    cpt_cloud = count_pts([test1, test2, test3, test4, test5, test6])  # Count of positive cloud tests
    cpt_clear = count_pts([test8, test9, test10, test11, test12, test13])  # Count of positive clear tests

    # Classify pixels based on test results and confidence
    choice = id_clear < id_cloud  # Choose cloud if cloud evidence stronger
    clear  = (id_cloud == 0) & (cpt_clear > 1)  # No cloud tests passed, multiple clear tests passed
    cloudy = ((id_clear == 0) & (cpt_cloud > 1)) | ((id_clear <= 1) & (cpt_cloud > 2))  # Strong cloud evidence

    # Assign confidence levels based on test results
    cloud_certain   = choice & cloudy  # Cloud choice with strong cloud evidence
    cloudy.values   = ~cloudy.values
    cloud_uncertain = choice & cloudy  # Cloud choice without strong evidence
    choice.values   = ~choice.values
    clear_certain   = choice & clear  # Clear choice with strong clear evidence
    clear.values    = ~clear.values
    clear_uncertain = choice & clear  # Clear choice without strong evidence

    # Combine confidence classifications into single land mask
    land_mask = label['cloud_certain']*cloud_certain + \
                label['cloud_uncertain']*cloud_uncertain + \
                label['clear_certain']*clear_certain + \
                label['clear_uncertain']*clear_uncertain
    
    # Merge sea and land masks based on water mask
    mask = where(water, sea_mask, land_mask)
    mask.attrs.update(label)  # Add label definitions to mask attributes
    
    return mask