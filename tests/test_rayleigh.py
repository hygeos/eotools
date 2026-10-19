import numpy as np
import pytest
import xarray as xr

from eotools.rayleigh import calc_odr


def make_ds(pressure_kind: str, P: float = 1013.25, z: float = 0.) -> xr.Dataset:
    shp = (3, 4)
    return xr.Dataset({
        "altitude": (("y", "x"), np.full(shp, z), {"units": "m"}),
        pressure_kind: (("y", "x"), np.full(shp, P), {"units": "hPa"}),
        "cwav": (("bands",), np.array([443., 865.], dtype="float32"), {"units": "nm"}),
    }, coords={"bands": [443, 865]})


@pytest.mark.parametrize("pressure_kind", ["surface_pressure", "sea_level_pressure"])
def test_calc_odr_cwav(pressure_kind):
    """
    At altitude 0, surface and sea-level pressures give the same ODR
    """
    odr = calc_odr(make_ds(pressure_kind))
    assert odr.sizes["bands"] == 2
    assert np.allclose(odr.sel(bands=443), 0.236, atol=1e-3)
