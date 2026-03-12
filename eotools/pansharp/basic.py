from core.process.blockwise import BlockProcessor
from core.interpolate import Interpolator, Nearest
from core.geo.naming import names, Var

from dataclasses import dataclass, field
from xarray import DataArray
from numpy import linspace


valid_ms_vars = [str(names.rtoa), str(names.ltoa), str(names.bt)]

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
    
    def process_block(self, block):
         
        multi = block[self.var_ms]
        panchro = block[self.panchro]
        maps = self.spatial_mapping
        
        # Modify coordinates before interpolation
        sizes = panchro.sizes
        dims = [str(names.rows), str(names.columns)]
        multi = multi.assign_coords({
            d: linspace(0, sizes[maps[d]]-1, multi.sizes[d]) for d in dims
        }).to_dataset(name='multi')
        panchro = panchro.assign_coords({
            maps[d]: linspace(0, sizes[maps[d]]-1, sizes[maps[d]]) for d in dims
        }).to_dataset(name='panchro')
        
        # Upscale multi spectral bands spatial resolution
        interp_dims = {d: Nearest(maps[d]) for d in dims}
        multi = Interpolator(multi, **interp_dims).map_blocks(panchro)['multi']
        
        # Compute pseudo-panchromatic
        if self.weights is not None: 
            weights = DataArray(self.weights, dims=str(names.bands))
            pseudopan = (weights*multi).sum(dim=str(names.bands))
            pseudopan /= weights.sum()
        else: 
            pseudopan = multi.mean(dim=str(names.bands))
        
        # Apply Brovey transformation
        ratio = panchro['panchro'] / pseudopan
        block['upsample'] = multi * ratio