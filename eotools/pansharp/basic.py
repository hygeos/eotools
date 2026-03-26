from core.process.blockwise import BlockProcessor
from core.interpolate import Interpolator, Nearest
from core.geo.naming import names, Var

from dataclasses import dataclass, field
from xarray import DataArray, zeros_like
from numpy import linspace
from scipy.ndimage import uniform_filter
import numpy as np


valid_ms_vars = [str(names.rtoa), str(names.ltoa), str(names.bt)]


def match_histogram(src: DataArray, ref: DataArray) -> DataArray:
    """Stretch src to match the mean and std of ref.
    
    Parameters
    ----------
    src : DataArray
        Source array to be stretched.
    ref : DataArray
        Reference array whose statistics will be matched.
    
    Returns
    -------
    DataArray
        Source array stretched to match reference statistics.
    """
    return (src - src.mean()) / (src.std() + 1e-6) * ref.std() + ref.mean()


def supersample_multi(multi: DataArray, pan: DataArray, mapping: dict) -> DataArray:
    sizes = pan.sizes
    dims = mapping.keys()
    multi = multi.assign_coords({
        d: linspace(0, sizes[mapping[d]]-1, multi.sizes[d]) for d in dims
    }).to_dataset(name='multi')
    pan = pan.assign_coords({
        mapping[d]: linspace(0, sizes[mapping[d]]-1, sizes[mapping[d]]) for d in dims
    }).to_dataset(name='panchro')
    
    # Upscale multi spectral bands spatial resolution
    interp_dims = {d: Nearest(mapping[d]) for d in dims}
    return Interpolator(multi, **interp_dims).map_blocks(pan)['multi']



@dataclass
class Brovey(BlockProcessor):
    """Brovey pansharpening transformation.
    
    The Brovey transform is a simple method for pansharpening that enhances the 
    spatial resolution of multispectral imagery by combining it with a higher 
    resolution panchromatic band.
    
    Attributes
    ----------
    var_ms : str
        Name of the multispectral variable to pansharpen. Must be one of the 
        valid multispectral variables (rtoa, ltoa, or bt).
    panchro : str
        Name of the panchromatic band variable used for sharpening.
    spatial_mapping : dict
        Mapping of multispectral spatial dimension names to panchromatic spatial 
        dimension names. Used to align and rename dimensions after upsampling.
        Example: {str(names.rows): 'pan_rows', str(names.columns): 'pan_cols'}
    weights : list or None, optional
        Band weights for computing the pseudo-panchromatic band. If None (default), 
        uses equal weights (simple mean). If provided, must have length equal to 
        the number of bands.
    
    Notes
    -----
    The Brovey transformation formula is:
    
        pansharp_i = MS_i * (Pan / PseudoPan)
    
    where:
        - MS_i is the upsampled multispectral band i
        - Pan is the panchromatic band
        - PseudoPan is the weighted mean (or simple mean) of all MS bands
        - pansharp_i is the resulting pansharpened band i
    
    Examples
    --------    
    >>> # With custom band weights
    >>> processor = Brovey(
    ...     var_ms='Rtoa',
    ...     panchro='pan',
    ...     spatial_mapping={'y': 'pan_y', 'x': 'pan_x'},
    ...     weights=[0.25, 0.5, 0.25]
    ... )
    """
    var_ms: str
    panchro: str
    spatial_mapping: dict
    weights: list|None = field(default=None)
    
    def input_vars(self):
        assert self.var_ms in valid_ms_vars
        return [Var(self.panchro), Var(self.var_ms)]
    
    def created_vars(self):
        return [Var('upsample')]
    
    def auto_template(self) -> bool:
        return True
    
    def process_block(self, block):
         
        multi = block[self.var_ms]
        panchro = block[self.panchro]
        maps = self.spatial_mapping
        
        # Upscale multi spectral bands spatial resolution
        multi = supersample_multi(multi, panchro, maps)
        
        # Compute pseudo-panchromatic
        if self.weights is not None: 
            weights = DataArray(self.weights, dims=str(names.bands))
            pseudopan = (weights*multi).sum(dim=str(names.bands))
            pseudopan /= weights.sum()
        else: 
            pseudopan = multi.mean(dim=str(names.bands))
        
        # Apply Brovey transformation
        ratio = panchro / pseudopan
        block['upsample'] = multi * ratio


@dataclass
class IHS(BlockProcessor):
    """IHS (Intensity–Hue–Saturation) pansharpening transformation.
    
    The IHS transform converts RGB bands to IHS color space, replaces the 
    Intensity channel with a histogram-matched panchromatic band, then converts 
    back to the original color space. This method is simple but restricted to 
    3 bands (RGB).
    
    Attributes
    ----------
    var_ms : str
        Name of the multispectral variable to pansharpen. Must be one of the 
        valid multispectral variables (rtoa, ltoa, or bt) and must have exactly 
        3 bands.
    panchro : str
        Name of the panchromatic band variable used for sharpening.
    spatial_mapping : dict
        Mapping of multispectral spatial dimension names to panchromatic spatial 
        dimension names. Used to align and rename dimensions after upsampling.
        Example: {str(names.rows): 'pan_rows', str(names.columns): 'pan_cols'}
    
    Notes
    -----
    The IHS transformation formula is:
    
        Intensity = (R + G + B) / 3
        Pan_matched = histogram_match(Pan, Intensity)
        delta = Pan_matched - Intensity
        R_sharp = R + delta
        G_sharp = G + delta
        B_sharp = B + delta
    
    where histogram matching stretches the panchromatic band to match the mean 
    and standard deviation of the computed intensity.
    
    This method is restricted to 3-band imagery (typically RGB).
    
    Examples
    --------    
    >>> processor = IHS(
    ...     var_ms='Rtoa',
    ...     panchro='pan',
    ...     spatial_mapping={'y': 'pan_y', 'x': 'pan_x'}
    ... )
    """
    var_ms: str
    panchro: str
    spatial_mapping: dict
    
    def input_vars(self):
        assert self.var_ms in valid_ms_vars
        return [Var(self.panchro), Var(self.var_ms)]
    
    def created_vars(self):
        return [Var('upsample')]
    
    def auto_template(self) -> bool:
        return True
    
    def process_block(self, block):
         
        multi = block[self.var_ms]
        panchro = block[self.panchro]
        maps = self.spatial_mapping
        
        # Validate that we have exactly 3 bands for IHS
        n_bands = multi.sizes[str(names.bands)]
        if n_bands != 3:
            raise ValueError(
                f"IHS pansharpening requires exactly 3 bands, got {n_bands}. "
                "IHS is restricted to RGB imagery."
            )
        
        # Upscale multi spectral bands spatial resolution
        multi = supersample_multi(multi, panchro, maps)
        
        # Compute intensity as mean of RGB bands
        intensity = multi.mean(dim=str(names.bands))
        
        # Histogram match panchromatic to intensity
        pan_matched = match_histogram(panchro, intensity)
        
        # Compute delta and inject into each band
        delta = pan_matched - intensity
        block['upsample'] = multi + delta


@dataclass
class HPF(BlockProcessor):
    """HPF (High-Pass Filter) injection pansharpening.
    
    Isolates high-frequency spatial detail from PAN using a low-pass filter, 
    then adds that detail into each MS band. Very controlled; spectral integrity 
    is largely preserved.
    
    Attributes
    ----------
    var_ms : str
        Name of the multispectral variable to pansharpen. Must be one of the 
        valid multispectral variables (rtoa, ltoa, or bt).
    panchro : str
        Name of the panchromatic band variable used for sharpening.
    spatial_mapping : dict
        Mapping of multispectral spatial dimension names to panchromatic spatial 
        dimension names. Used to align and rename dimensions after upsampling.
        Example: {str(names.rows): 'pan_rows', str(names.columns): 'pan_cols'}
    kernel_size : int, optional
        Size of the low-pass filter kernel. Default is 5.
    weights : list or None, optional
        Per-band injection weights. If None (default), uses equal weights (1.0) 
        for all bands. If provided, must have length equal to the number of bands.
    
    Notes
    -----
    The HPF transformation process:
    
    1. Apply low-pass filter to panchromatic band
    2. Extract high-frequency detail: hpf = pan - lpf
    3. Inject detail into each MS band: ms_sharp = ms + weight * hpf
    
    This method preserves spectral integrity well since it only adds spatial 
    detail without modifying the spectral ratios.
    
    Examples
    --------    
    >>> processor = HPF(
    ...     var_ms='Rtoa',
    ...     panchro='pan',
    ...     spatial_mapping={'y': 'pan_y', 'x': 'pan_x'},
    ...     kernel_size=5,
    ...     weights=[1.0, 1.0, 1.0]
    ... )
    """
    var_ms: str
    panchro: str
    spatial_mapping: dict
    kernel_size: int = 5
    weights: list|None = field(default=None)
    
    def input_vars(self):
        assert self.var_ms in valid_ms_vars
        return [Var(self.panchro), Var(self.var_ms)]
    
    def created_vars(self):
        return [Var('upsample')]
    
    def auto_template(self) -> bool:
        return True
    
    def process_block(self, block):
         
        multi = block[self.var_ms]
        panchro = block[self.panchro]
        maps = self.spatial_mapping
        
        # Upscale multi spectral bands spatial resolution
        multi = supersample_multi(multi, panchro, maps)
        
        # Apply low-pass filter to panchromatic
        lpf = uniform_filter(panchro.astype(float), size=self.kernel_size)
        
        # Extract high-frequency detail
        hpf = zeros_like(panchro)
        hpf.data = panchro.values - lpf
        
        # Apply weights if provided
        if self.weights is not None:
            weights = DataArray(self.weights, dims=str(names.bands))
            # Inject weighted detail into each band
            block['upsample'] = multi + weights * hpf
        else:
            # Inject detail with equal weight into each band
            block['upsample'] = multi + hpf