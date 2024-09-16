from typing import Dict, Optional
import xarray as xr
from core.interpolate import interp
from eoread.utils.xrtags import tag_add
import pint_xarray     # noqa: F401
from cf_xarray.units import units
from core.tools import datetime

units.define('Dobson = 2.1415E-05 kg/m**2')


def apply_ancillary(
    ds: xr.Dataset,
    ancillary,
    variables: Dict,
    interp_dims: Optional[Dict]=None,
    overwrite: Optional[bool] = None,
    tag: Optional[str] = "ancillary",
    **kwargs,
):
    """
    Apply ancillary data from `ancillary` to dataset `ds`

    Arguments:

        ds: destination dataset

        ancillary: ancillary data provider

        variables: dictionary containing the list of variables to get and
            interpolate in `ancillary`, and their units (as interpreted
            by the `pint` library).
            Example:
            {
                'horizontal_wind': 'm/s',
                'sea_level_pressure': 'hectopascals',
                'total_column_ozone': 'Dobson',
            }
        
        interp_dims: maps the dimensions in `ancillary` to the variables in `ds`.
            Example:
            {"latitude": "latitude", "longitude": "longitude"}

        overwrite: whether to overwrite existing variables
            (True, False, None: raise an Error in case of existing variable)
        
        tag: Defines the tag to be added to the output variables.
    """
    interp_dims = interp_dims or {"latitude": "latitude", "longitude": "longitude"}

    anc = ancillary.get(datetime(ds))
    
    # convert the ancillary data to the desired units
    anc = anc.pint.quantify().pint.to(variables).pint.dequantify()
    
    for varname in variables:
        if varname in ds:
            if overwrite is None:
                raise RuntimeError(f"Error - would not overwrite {varname}")
            elif not overwrite:
                continue

        ds[varname] = interp(
            anc[varname].compute(scheduler="sync"),
            interp={
                k: ds[v] for (k, v) in interp_dims.items()
            },
        )
        if tag is not None:
            tag_add(ds[varname], tag)
