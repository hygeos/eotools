
import xarray as xr
import pytest
from eotools import units


@pytest.mark.parametrize("input_unit,output_unit", [
    ('kg/m²', 'Dobson'),
    ('Dobson', 'kg m**-2'),
    ('m2', 'cm2'),
    ('s', 'day'),
    ('m/s', 'knot'),
])
def test_units_conversion(input_unit, output_unit):
    def sample(unit: str):
        da = xr.DataArray(
            1, 
            attrs={'units': unit}
        )
        return da
    result = units.convert(sample(input_unit), output_unit)
    print(result)
