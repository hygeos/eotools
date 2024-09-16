import numpy as np
import xarray as xr
from core.tools import raiseflag

from eotools.utils.stats import stdNxN

"""
Basic cloud mask, as implemented in Polymer
"""


class Cloud_mask:
    """
    Cloud masking module

    Cloud_mask(ds).apply()
    """

    def __init__(
        self,
        ds: xr.Dataset,
        cm_input_var: str,
        cm_band_nir: int,
        cm_flag_value: int,
        cm_flag_name: str = "CLOUD",
        cm_flag_variable: str = "flags",
        cm_thres_Rcloud=0.2,
        cm_thres_Rcloud_std=0.04,
        cm_dist: int = 3,
        bitmask_invalid: int = -1,
        **kwargs,
    ):
        self.ds = ds
        self.thres_Rcloud = cm_thres_Rcloud
        self.thres_Rcloud_std = cm_thres_Rcloud_std
        self.band_nir = cm_band_nir
        self.dist = cm_dist
        self.input_var = cm_input_var
        self.flag_value = cm_flag_value
        self.flag_name = cm_flag_name
        self.flag_variable = cm_flag_variable
        self.bitmask_invalid = bitmask_invalid

    def run(self, Rnir, flags):
        ok = (flags & self.bitmask_invalid) == 0
        cloudmask = np.zeros_like(flags, dtype="uint8")
        cloudmask[:] = Rnir > self.thres_Rcloud
        cloudmask |= stdNxN(Rnir, self.dist, ok, fillv=0.0) > self.thres_Rcloud_std

        return cloudmask

    def apply(self):
        ds = self.ds
        Rnir = ds[self.input_var].sel(bands=self.band_nir)

        cloudmask = xr.apply_ufunc(
            self.run,
            Rnir,
            ds[self.flag_variable],
            dask="parallelized",
            input_core_dims=[[], []],
            output_core_dims=[[]],
            output_dtypes=["uint8"],
        )

        raiseflag(ds[self.flag_variable], self.flag_name, self.flag_value, cloudmask)
