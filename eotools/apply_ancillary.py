from datetime import datetime
from typing import Dict, Optional
import xarray as xr
from core.interpolate import interp, Linear, Interpolator
from core.xrtags import tag_add
import pint_xarray     # noqa: F401
from cf_xarray.units import units
from core.tools import datetime as datetime_parse

units.define('Dobson = 2.1415E-05 kg/m**2')

class ApplyAncillary(Interpolator):
    """Interpolator for ancillary data retrieved from a provider at a given datetime.

    This class extends Interpolator to handle ancillary data that varies with time,
    by fetching the appropriate data from an ancillary provider for the specified datetime.

    Parameters
    ----------
    dt : datetime | xr.DataArray
        The datetime or dataset to retrieve ancillary data for. If a dataset is provided,
        its datetime will be extracted using `core.tools.datetime`.
    ancillary_provider : object
        An ancillary data provider object with a `get()` method that accepts a datetime
        and returns an xarray Dataset containing ancillary variables.

    Examples
    --------
    >>> from datetime import datetime
    >>> from eoread.ancillary_nasa import Ancillary_NASA
    >>> processor = ApplyAncillary(datetime(2023, 1, 1), Ancillary_NASA())
    >>> result = processor.map_blocks(dataset)
    ... # result contains all variables provided by Ancillary_NASA(), interpolated
    ... # over the variables provided by `dataset`.
    """
    def __init__(
        self,
        dt: datetime | xr.Dataset,
        ancillary_provider,
    ):
        if isinstance(dt, xr.Dataset):
            dt = datetime_parse(dt)
        assert isinstance(dt, datetime)
        data = ancillary_provider.get(dt)
        super().__init__(
            data,
            latitude=Linear("latitude"),
            # TODO: check whether we can apply cyclic indexing on longitude
            longitude=Linear("longitude"),
        )


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

        ancillary: ancillary data provider. If None, just check that all variables are
            already provided in ds.

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
    # TODO: deprecated. Please use the BlockProcessor "AncillaryProcessor"
    if ancillary is None:
        # check that all variables are provided in ds with the expected unit
        for var in variables.keys():
            assert var in ds
            if units(ds[var].units).units != units(variables[var]).units:
                ds[var] = ds[var].pint.quantify().pint.to(units(variables[var]).units).pint.dequantify()
        
        return

    interp_dims = interp_dims or {"latitude": "latitude", "longitude": "longitude"}

    anc = ancillary.get(datetime_parse(ds))
    
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
            **{
                k: Linear(ds[v]) for (k, v) in interp_dims.items()
            },
        )
        if tag is not None:
            tag_add(ds[varname], tag)
