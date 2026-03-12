import numpy as np
import xarray as xr

from core.geo.naming import names
from matplotlib import pyplot as plt
from core.tests.conftest import savefig
from eotools.pansharp.basic import valid_ms_vars


class _Test:
    
    @staticmethod
    def test_real_image_pansharpening(processor, image, request):
        """Test HPF pansharpening on the Hygeos logo."""        
        var_ms = valid_ms_vars[0]
        
        # Degrade resolution by factor of 4
        degradation_factor = 30
        degraded = image.thin(y=degradation_factor, x=degradation_factor)
        
        # Create panchromatic as weighted mean of RGB
        pan_weights = [0.25, 0.6, 0.15]  # Typical pan weights
        pan_reference = np.average(image, axis=2, weights=pan_weights)
        
        # Create xarray data structures
        pan_data = xr.DataArray(pan_reference, dims=['pan_y', 'pan_x'])
        
        block = {'pan': pan_data, var_ms: degraded}
        processor.process_block(block)
        result_da = block['upsample']
        
        # Convert to numpy array with proper dimension order (H, W, C)
        result = result_da.transpose('pan_y', 'pan_x', str(names.bands)).values
        
        # Create visualization with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        # Reference (original high-res)
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Reference (Original HR)')
        axes[0, 0].axis('off')
        
        # Degraded (low-res, upsampled for visualization)
        axes[0, 1].imshow(degraded)
        axes[0, 1].set_title(f'Degraded')
        axes[0, 1].axis('off')
        
        # Pansharpened result
        axes[1, 0].imshow(result)
        axes[1, 0].set_title('Brovey Pansharpened')
        axes[1, 0].axis('off')
        
        # Panchromatic band
        axes[1, 1].imshow(pan_reference, cmap='gray')
        axes[1, 1].set_title('Panchromatic Band')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        savefig(request)
        plt.close()
        
        # Verify output shape matches pan
        assert result.shape == image.shape