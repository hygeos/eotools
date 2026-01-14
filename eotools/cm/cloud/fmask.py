import numpy as np
from xarray import DataArray


def FMASK(
    blue: DataArray,
    green: DataArray,
    red: DataArray,
    nir: DataArray,
    swir1: DataArray,
    swir2: DataArray,
    cirrus: DataArray,
    tir1: DataArray,
    tir2: DataArray,
) -> tuple[DataArray, DataArray, DataArray]:
    """Fmask cloud, cloud shadow, and water detection algorithm.
    
    Implements the Function of mask (Fmask) algorithm for automated detection
    of clouds, cloud shadows, and water in Landsat imagery. The algorithm uses
    multi-spectral tests, temperature-based probability assessments, and spatial
    filtering to produce robust cloud masks.
    
    Parameters
    ----------
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
    tir1 : DataArray
        Brightness temperature of thermal infrared band at 10.8 µm in Kelvin
    tir2 : DataArray
        Brightness temperature of thermal infrared band at 12 µm in Kelvin
    
    Returns
    -------
    pcloud : DataArray
        Boolean cloud mask where True indicates cloud pixels
    pshadow : DataArray
        Boolean cloud shadow mask where True indicates cloud shadow pixels
    water : DataArray
        Boolean water mask where True indicates water pixels
    
    References
    ----------
    .. [1] Zhu, Z. and Woodcock, C.E., 2012. Object-based cloud and cloud shadow
           detection in Landsat imagery. Remote sensing of environment, 118,
           pp.83-94.
    .. [2] Zhu, Z. and Woodcock, C.E., 2015. Improvement and expansion of the
           Fmask algorithm: cloud, cloud shadow, and snow detection for Landsats
           4–7, 8, and Sentinel 2 images. Remote Sensing of Environment, 159,
           pp.269-277.
    """
    # Call original FMask algorithm with required bands
    return _FmaskCM(
        blue,
        green,
        red,
        nir,
        swir1,
        swir2,
        cirrus,
        tir1,
        tir2,
        np.ones_like(blue, dtype=bool)
    ).cloud_mask()


class _FmaskCM(object):
    ''' Implement fmask algorithm.
    :param image: Landsat sat_image stack LandsatImage object
    :return: fmask object
    '''

    def __init__(self, blue, green, red, nir, swir1, swir2, cirrus, tirs1, tirs2, mask):

        self.image = None
        self.shape = mask.shape
        self.mask = mask > 0.
        self.sat = 'LC8'

        self.blue = blue
        self.green = green
        self.red = red
        self.nir = nir
        self.swir1 = swir1
        self.swir2 = swir2
        self.cirrus = cirrus
        self.tirs1 = tirs1 - 273.15
        self.tirs2 = tirs2 - 273.15

        self.ndvi = (self.nir-self.red)/(self.nir+self.red)
        self.ndsi = (self.green-self.swir1)/(self.green+self.swir1)


    def basic_test(self):
        """Fundamental test to identify Potential Cloud Pixels (PCPs)
        Equation 1 (Zhu and Woodcock, 2012)
        Note: all input arrays must be the same shape
        Parameters
        ----------
        ndvi: ndarray
        ndsi: ndarray
        swir2
            Shortwave Infrared Band TOA reflectance
            Band 7 in Landsat 8, ~2.2 µm
        tirs1: ndarray
            Thermal band brightness temperature
            Band 10 in Landsat 8, ~11 µm
            units are degrees Celcius
        Output
        ------
        ndarray: boolean
        """
        # Thresholds
        th_ndsi = 0.8  # index
        th_ndvi = 0.8  # index
        th_tirs1 = 27.0  # Celsius
        th_swir2 = 0.03  # toa
        return ((self.swir2 > th_swir2) &
                (self.tirs1 < th_tirs1) &
                (self.ndsi < th_ndsi) &
                (self.ndvi < th_ndvi))

    def whiteness_index(self):
        """Index of "Whiteness" based on visible bands.
        Parameters
        ----------

        Output
        ------
        ndarray:
            whiteness index
        """
        mean_vis = (self.blue + self.green + self.red) / 3

        blue_absdiff = np.absolute(self._divide_zero(self.blue - mean_vis, mean_vis))
        green_absdiff = np.absolute(self._divide_zero(self.green - mean_vis, mean_vis))
        red_absdiff = np.absolute(self._divide_zero(self.red - mean_vis, mean_vis))

        return blue_absdiff + green_absdiff + red_absdiff

    def whiteness_test(self):
        """Whiteness test
        Clouds appear white due to their "flat" reflectance in the visible bands
        Equation 2 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        blue: ndarray
        green: ndarray
        red: ndarray
        Output
        ------
        ndarray: boolean
        """
        whiteness_threshold = 0.7
        test = self.whiteness_index() < whiteness_threshold
        return test

    def hot_test(self):
        """Haze Optimized Transformation (HOT) test
        Equation 3 (Zhu and Woodcock, 2012)
        Based on the premise that the visible bands for most land surfaces
        are highly correlated, but the spectral response to haze and thin cloud
        is different between the blue and red wavelengths.
        Zhang et al. (2002)
        Parameters
        ----------
        blue: ndarray
        red: ndarray
        Output
        ------
        ndarray: boolean
        """
        thres = 0.08
        return self.blue - (0.5 * self.red) - thres > 0.0

    def nirswir_test(self):
        """Spectral test to exclude bright rock and desert
        see (Irish, 2000)
        Equation 4 (Zhu and Woodcock, 2012)
        Note that Zhu and Woodcock 2015 refer to this as the "B4B5" test
        due to the Landsat ETM+ band designations. In Landsat 8 OLI,
        these are bands 5 and 6.
        Parameters
        ----------
        nir: ndarray
        swir1: ndarray
        Output
        ------
        ndarray: boolean
        """
        th_ratio = 0.75

        return (self.nir / self.swir1) > th_ratio

    def cirrus_test(self):
        """Cirrus TOA test, see (Zhu and Woodcock, 2015)
        The threshold is derived from (Wilson & Oreopoulos, 2013)
        Parameters
        ----------
        cirrus: ndarray
        Output
        ------
        ndarray: boolean
        """
        th_cirrus = 0.0113

        return self.cirrus > th_cirrus

    def water_test(self):
        """Water or Land?
        Equation 5 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        ndvi: ndarray
        nir: ndarray
        Output
        ------
        ndarray: boolean
        """
        th_ndvi_A = 0.01
        th_nir_A = 0.11
        th_ndvi_B = 0.1
        th_nir_B = 0.05

        return (((self.ndvi < th_ndvi_A) & (self.nir < th_nir_A)) |
                ((self.ndvi < th_ndvi_B) & (self.nir < th_nir_B)))

    def potential_cloud_pixels(self):
        """Determine potential cloud pixels (PCPs)
        Combine basic spectral testsr to get a premliminary cloud mask
        First pass, section 3.1.1 in Zhu and Woodcock 2012
        Equation 6 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        ndvi: ndarray
        ndsi: ndarray
        blue: ndarray
        green: ndarray
        red: ndarray
        nir: ndarray
        swir1: ndarray
        swir2: ndarray
        cirrus: ndarray
        tirs1: ndarray
        Output
        ------
        ndarray:
            potential cloud mask, boolean
        """
        eq1 = self.basic_test()
        eq2 = self.whiteness_test()
        eq3 = self.hot_test()
        eq4 = self.nirswir_test()
        if self.sat == 'LC8':
            cir = self.cirrus_test()
            return (eq1 & eq2 & eq3 & eq4) | cir
        else:
            return eq1 & eq2 & eq3 & eq4

    def temp_water(self):
        """Use water to mask tirs and find 82.5 pctile
        Equation 7 and 8 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        is_water: ndarray, boolean
            water mask, water is True, land is False
        swir2: ndarray
        tirs1: ndarray
        Output
        ------
        float:
            82.5th percentile temperature over water
        """
        # eq7
        th_swir2 = 0.03
        water = self.water_test()
        clear_sky_water = water & (self.swir2 < th_swir2)

        # eq8
        clear_water_temp = self.tirs1.copy()
        clear_water_temp = np.where(clear_sky_water & self.mask, clear_water_temp, np.nan)
        pctl_clwt = np.nanpercentile(clear_water_temp, 82.5)
        return pctl_clwt

    def water_temp_prob(self):
        """Temperature probability for water
        Equation 9 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        water_temp: float
            82.5th percentile temperature over water
        swir2: ndarray
        tirs1: ndarray
        Output
        ------
        ndarray:
            probability of cloud over water based on temperature
        """
        temp_const = 4.0  # degrees C
        water_temp = self.temp_water()
        return (water_temp - self.tirs1) / temp_const

    def brightness_prob(self, clip=True):
        """The brightest water may have Band 5 reflectance
        as high as LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF.11
        Equation 10 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        nir: ndarray
        clip: boolean
        Output
        ------
        ndarray:
            brightness probability, constrained LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF..1
        """
        thresh = 0.11
        bp = np.minimum(thresh, self.nir) / thresh
        if clip:
            bp= np.clip(bp, 0, 1)
        return bp

    def temp_land(self, pcps, water):
        """Derive high/low percentiles of land temperature
        Equations 12 an 13 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        pcps: ndarray
            potential cloud pixels, boolean
        water: ndarray
            water mask, boolean
        tirs1: ndarray
        Output
        ------
        tuple:
            17.5 and 82.5 percentile temperature over clearsky land
        """
        # eq 12
        clearsky_land = ~(pcps | water)

        # use clearsky_land to mask tirs1
        clear_land_temp = self.tirs1.copy()
        clear_land_temp = np.where(clearsky_land & self.mask, clear_land_temp, np.nan)

        # take 17.5 and 82.5 percentile, eq 13
        low, high = np.nanpercentile(clear_land_temp, (17.5, 82.5))
        return low, high

    def land_temp_prob(self, tlow, thigh):
        """Temperature-based probability of cloud over land
        Equation 14 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        tirs1: ndarray
        tlow: float
            Low (17.5 percentile) temperature of land
        thigh: float
            High (82.5 percentile) temperature of land
        Output
        ------
        ndarray :
            probability of cloud over land based on temperature
        """
        temp_diff = 4  # degrees
        return (thigh + temp_diff - self.tirs1) / (thigh + 4 - (tlow - 4))

    def variability_prob(self, whiteness):
        """Use the probability of the spectral variability
        to identify clouds over land.
        Equation 15 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        ndvi: ndarray
        ndsi: ndarray
        whiteness: ndarray
        Output
        ------
        ndarray :
            probability of cloud over land based on variability
        """

        if self.sat in ['LT5', 'LE7']:
            # check for green and red saturation

            # if red is saturated and less than nir, ndvi = LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF
            mod_ndvi = np.where(self.red_saturated & (self.nir > self.red), 0, self.ndvi)

            # if green is saturated and less than swir1, ndsi = LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF
            mod_ndsi = np.where(self.green_saturated & (self.swir1 > self.green), 0, self.ndsi)
            ndi_max = np.fmax(np.absolute(mod_ndvi), np.absolute(mod_ndsi))

        else:
            ndi_max = np.fmax(np.absolute(self.ndvi), np.absolute(self.ndsi))

        f_max = 1.0 - np.fmax(ndi_max, whiteness)

        return f_max

    def land_threshold(self, land_cloud_prob, pcps, water):
        """Dynamic threshold for determining cloud cutoff
        Equation 17 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        land_cloud_prob: ndarray
            probability of cloud over land
        pcps: ndarray
            potential cloud pixels
        water: ndarray
            water mask
        Output
        ------
        float:
            land cloud threshold
        """
        # eq 12
        clearsky_land = ~(pcps | water)

        # 82.5th percentile of lCloud_Prob(masked by clearsky_land) + LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF.2
        cloud_prob = land_cloud_prob.copy()
        cloud_prob = np.where(clearsky_land & self.mask, cloud_prob, np.nan)

        # eq 17
        th_const = 0.2
        return np.nanpercentile(cloud_prob, 82.5) + th_const

    def potential_cloud_layer(self, pcp, water, tlow, land_cloud_prob, land_threshold,
            water_cloud_prob, water_threshold=0.5):
        """Final step of determining potential cloud layer
        Equation 18 (Zhu and Woodcock, 2012)
        
        Saturation (green or red) test is not in the algorithm
        
        Parameters
        ----------
        pcps: ndarray
            potential cloud pixels
        water: ndarray
            water mask
        tirs1: ndarray
        tlow: float
            low percentile of land temperature
        land_cloud_prob: ndarray
            probability of cloud over land
        land_threshold: float
            cutoff for cloud over land
        water_cloud_prob: ndarray
            probability of cloud over water
        water_threshold: float
            cutoff for cloud over water
        Output
        ------
        ndarray:
            potential cloud layer, boolean
        """
        # Using pcp and water as mask todo
        # change water threshold to dynamic, line 132 in Zhu, 2015 todo
        part1 = (pcp & water & (water_cloud_prob > water_threshold))
        part2 = (pcp & ~water & (land_cloud_prob > land_threshold))
        temptest = self.tirs1 < (tlow - 35)  # 35degrees C colder

        if self.sat in ['LT5', 'LE7']:
            saturation = self.blue_saturated | self.green_saturated | self.red_saturated

            return part1 | part2 | temptest | saturation

        else:
            return part1 | part2 | temptest

    def potential_cloud_shadow_layer(self, water):
        """Find low NIR/SWIR1 that is not classified as water
        This differs from the Zhu Woodcock algorithm
        but produces decent results without requiring a flood-fill
        Parameters
        ----------
        nir: ndarray
        swir1: ndarray
        water: ndarray
        Output
        ------
        ndarray
            boolean, potential cloud shadows
        """
        return (self.nir < 0.10) & (self.swir1 < 0.10) & ~water

    def potential_snow_layer(self):
        """Spectral test to determine potential snow
        Uses the 9.85C (283K) threshold defined in Zhu, Woodcock 2015
        Parameters
        ----------
        ndsi: ndarray
        green: ndarray
        nir: ndarray
        tirs1: ndarray
        Output
        ------
        ndarray:
            boolean, True is potential snow
        """
        return (self.ndsi > 0.15) & (self.tirs1 < 9.85) & (self.nir > 0.11) & (self.green > 0.1)

    def cloud_mask(self, min_filter=(3, 3), max_filter=(10, 10), combined=False, cloud_and_shadow=False):
        """Calculate the potential cloud layer from source data
        *This is the high level function which ties together all
        the equations for generating potential clouds*
        Parameters
        ----------
        blue: ndarray
        green: ndarray
        red: ndarray
        nir: ndarray
        swir1: ndarray
        swir2: ndarray
        cirrus: ndarray
        tirs1: ndarray
        min_filter: 2-element tuple, default=(3,3)
            Defines the window for the minimum_filter, for removing outliers
        max_filter: 2-element tuple, default=(21, 21)
            Defines the window for the maximum_filter, for "buffering" the edges
        combined: make a boolean array masking all (cloud, shadow, water)
        Output
        ------
        ndarray, boolean:
            potential cloud layer; True = cloud
        ndarray, boolean
            potential cloud shadow layer; True = cloud shadow
            :param cloud_and_shadow:
        """
        # logger.info("Running initial testsr")
        whiteness = self.whiteness_index()
        water = self.water_test()

        # First pass, potential clouds
        pcps = self.potential_cloud_pixels()

        if self.sat == 'LC8':
            cirrus_prob = self.cirrus / 0.04
        else:
            cirrus_prob = 0.0

        # Clouds over water
        wtp = self.water_temp_prob()
        bp = self.brightness_prob()
        water_cloud_prob = (wtp * bp) + cirrus_prob
        wthreshold = 0.5

        # Clouds over land
        tlow, thigh = self.temp_land(pcps, water)
        ltp = self.land_temp_prob(tlow, thigh)
        vp = self.variability_prob(whiteness)
        land_cloud_prob = (ltp * vp) + cirrus_prob
        lthreshold = self.land_threshold(land_cloud_prob, pcps, water)

        # logger.info("Calculate potential clouds")
        pcloud = self.potential_cloud_layer(
            pcps, water, tlow,
            land_cloud_prob, lthreshold,
            water_cloud_prob, wthreshold)

        # Ignoring snow for now as it exhibits many false positives and negatives
        # when used as a binary mask
        # psnow = potential_snow_layer(ndsi, green, nir, tirs1)
        # pcloud = pcloud & ~psnow

        # logger.info("Calculate potential cloud shadows")
        pshadow = self.potential_cloud_shadow_layer(water)

        # The remainder of the algorithm differs significantly from Fmask
        # In an attempt to make a more visually appealling cloud mask
        # with fewer inclusions and more broad shapes

        if min_filter:
            # Remove outliers
            # logger.info("Remove outliers with minimum filter")

            from scipy.ndimage.filters import minimum_filter
            from scipy.ndimage.morphology import distance_transform_edt

            # remove cloud outliers by nibbling the edges
            pcloud = minimum_filter(pcloud, size=min_filter)

            # crude, just look x pixels away for potential cloud pixels
            dist = distance_transform_edt(~pcloud)
            pixel_radius = 100.0
            pshadow = (dist < pixel_radius) & pshadow

            # remove cloud shadow outliers
            pshadow = minimum_filter(pshadow, size=min_filter)

        if max_filter:
            # grow around the edges
            # logger.info("Buffer edges with maximum filter")

            from scipy.ndimage.filters import maximum_filter

            pcloud = maximum_filter(pcloud, size=max_filter)
            pshadow = maximum_filter(pshadow, size=max_filter)

        # mystery, save pcloud here, shows no nan in qgis, save later, shows nan
        # outfile = '/data01/images/sandbox/pcloud.tif'
        # georeference = self.sat_image.rasterio_geometry
        # array = pcloud
        # array = array.reshape(1, array.shape[LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF], array.shape[1])
        # array = np.array(array, dtype=georeference['dtype'])
        # with rasterio.open(outfile, 'w', **georeference) as dst:
        #     dst.write(array)
        # mystery test
        if combined:
            return pcloud | pshadow | water

        if cloud_and_shadow:
            return pcloud | pshadow

        return pcloud, pshadow, water

    @staticmethod
    def _divide_zero(a, b, replace=0):
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c = np.where(c == np.inf, replace, c)
            c = np.nan_to_num(c)
            return c