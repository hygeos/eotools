import xarray as xr
import pint_xarray  # noqa: F401
from cf_xarray.units import units


units.define("Dobson = 2.1415E-05 kg/m**2")


def convert(da: xr.DataArray, new_units: str) -> xr.DataArray:
    """
    Use pint-xarray for units conversion

    Example:
        # Detect the input data units (through the 'units' attribute) and
        # convert, if necessary, to 'Dobson'
        convert(total_ozone, 'Dobson')
    """
    # Check if units are equivalent in pint sense to avoid unnecessary conversion
    current_unit = da.attrs.get("units")
    if current_unit and (units(current_unit) == units(new_units)):
        return da

    return da.pint.quantify().pint.to(new_units).pint.dequantify()
