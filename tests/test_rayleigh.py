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


@pytest.mark.parametrize("pressure_kind", ["surface_pressure", "sea_level_pressure"])
def test_calc_odr_custom(pressure_kind):
    """
    Custom ODR at altitude 0: scaled by the pressure
    """
    odr_custom = {443: 0.2367, 865: 0.0155}
    ds = make_ds(pressure_kind, P=1000.)
    odr = calc_odr(ds, odr=odr_custom, odr_pressure=1013.)
    for b, v in odr_custom.items():
        assert np.allclose(odr.sel(bands=b), v*1000./1013.)


def test_calc_odr_custom_altitude():
    """
    Custom ODR above sea level: the sea-level pressure is scaled to the
    altitude, the surface pressure is used as is
    """
    odr_custom = {443: 0.2367, 865: 0.0155}
    odr_slp = calc_odr(make_ds("sea_level_pressure", z=1000.), odr=odr_custom)
    odr_sp = calc_odr(make_ds("surface_pressure", z=1000.), odr=odr_custom)
    assert (odr_slp < 0.9*odr_sp).all()
    assert np.allclose(odr_sp.sel(bands=443), 0.2367)
