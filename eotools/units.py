import xarray as xr
import pint_xarray  # noqa: F401
from cf_xarray.units import units


units.define("Dobson = 2.1415E-05 kg/m**2")


def check_units(da: xr.DataArray, expected_units: str):
    """
    Validate that a DataArray has the expected units.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to check for units.
    expected_units : str
        The expected units string (e.g., 'Dobson', 'kg/m**2').
    """
    current_unit = da.attrs.get("units")
    if not current_unit:
        raise ValueError(f'No units defined for {da.name}')
    if units(current_unit) != units(expected_units):
        raise ValueError(
            f"Wrong units detected for {da.name}. Expected {expected_units}, found {current_unit}."
        )

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
