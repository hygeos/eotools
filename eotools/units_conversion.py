import xarray as xr
import pint_xarray  # noqa: F401
import pint
# TODO: check if cf_xarray is useful here

ureg = pint.UnitRegistry(force_ndarray_like=True)
ureg.define('Dobson = 2.1415E-05 kg/m**2')


def convert(da: xr.DataArray, new_units: str) -> xr.DataArray:
    """
    Use pint-xarray for units conversion
    """
    # Check if units are equivalent in pint sense to avoid unnecessary conversion
    current_unit = da.attrs.get('units')
    if current_unit and (ureg.Unit(current_unit) == ureg.Unit(new_units)):
        return da

    return da.pint.quantify(unit_registry=ureg).pint.to(new_units).pint.dequantify()