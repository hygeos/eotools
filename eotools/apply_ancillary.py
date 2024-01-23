from typing import Optional
import xarray as xr
from eoread.utils.interpolate import interp
from eoread.utils.xrtags import tag_add


def apply_ancillary(
    ds: xr.Dataset,
    ancillary: xr.Dataset,
    overwrite: Optional[bool] = None,
    tag: Optional[str] = "ancillary",
    **kwargs,
):
    """
    Apply ancillary data from `ancillary` to dataset `ds`

    `overwrite`: whether to overwrite existing variables
        (True, False, None: raise an Error in case of existing variable)
    """
    for varname in ancillary:
        if varname in ds:
            if overwrite is None:
                raise RuntimeError(f"Error - would not overwrite {varname}")
            elif not overwrite:
                continue

        ds[varname] = interp(
            ancillary[varname].compute(scheduler="sync"),
            ds,
            {"latitude": "latitude", "longitude": "longitude"},
        )
        if tag is not None:
            tag_add(ds[varname], tag)
