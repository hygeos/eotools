from typing import Any

import numpy as np
import xarray as xr
from core.process.blockwise import BlockProcessor
from core.tools import Var

from eotools.utils.stats import stdNxN

"""
Basic cloud mask, as implemented in Polymer
"""


class Cloud_mask(BlockProcessor):
    """
    Cloud masking using NIR reflectance threshold and spatial variability.

    Two criteria (OR):
    1. High reflectance: Rnir > threshold (cloud)
    2. High spatial variability: local std dev > threshold (cloud edge)
    """

    def __init__(
        self,
        cm_input_var: str,
        cm_band_nir: Any,
        cm_flag_value: int,
        cm_flag_name: str = "CLOUD",
        cm_flag_variable: str = "flags",
        cm_thres_Rcloud: float = 0.2,
        cm_thres_Rcloud_std: float = 0.04,
        cm_dist: int = 3,
        bitmask_invalid: int = -1,
        **kwargs,
    ):
        self.input_var = cm_input_var
        self.band_nir = cm_band_nir
        self.flag_value = cm_flag_value
        self.flag_name = cm_flag_name
        self.flag_variable = cm_flag_variable
        self.thres_Rcloud = cm_thres_Rcloud
        self.thres_Rcloud_std = cm_thres_Rcloud_std
        self.dist = cm_dist
        self.bitmask_invalid = bitmask_invalid

    def input_vars(self) -> list[Var]:
        return [
            Var(self.input_var),
        ]

    def modified_vars(self) -> list[Var]:
        return [
            Var(self.flag_variable, flags={self.flag_name: self.flag_value}),
        ]

    def process_block(self, block: xr.Dataset) -> None:
        Rnir = block[self.input_var].sel(bands=self.band_nir)
        flags = block[self.flag_variable]

        if self.bitmask_invalid >= 0:
            ok = (flags.values & self.bitmask_invalid) == 0
        else:
            ok = (flags.values >= 0)

        cloudmask = np.zeros_like(flags.values, dtype="uint8")
        cloudmask[:] = Rnir.values > self.thres_Rcloud
        cloudmask |= stdNxN(Rnir.values, self.dist, ok, fillv=0.0) > self.thres_Rcloud_std

        self.raiseflag(
            block,
            self.flag_variable,
            self.flag_name,
            xr.DataArray(cloudmask, dims=flags.dims),
        )
