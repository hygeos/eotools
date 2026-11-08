import numpy as np
import pytest
import xarray as xr

from eotools.rayleigh import RayleighCorrection, calc_odr


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


def make_block_rayleigh(muv_bands: np.ndarray, raa_bands: np.ndarray) -> xr.Dataset:
    """
    Block for RayleighCorrection, with per-band viewing geometry
    `muv`, `raa` (bands, y, x)
    """
    nb, ny, nx = muv_bands.shape
    shp = (ny, nx)
    bands = [443, 865][:nb]
    flags = np.zeros(shp, dtype="uint16")
    flags[0, 0] = 1  # one invalid pixel
    return xr.Dataset({
        "altitude": (("y", "x"), np.zeros(shp, dtype="float32"), {"units": "m"}),
        "sea_level_pressure": (("y", "x"), np.full(shp, 1013.25, dtype="float32"), {"units": "hPa"}),
        "horizontal_wind": (("y", "x"), np.full(shp, 5., dtype="float32"), {"units": "m/s"}),
        "mus": (("y", "x"), np.full(shp, 0.8, dtype="float32")),
        "muv": (("bands", "y", "x"), muv_bands.astype("float32")),
        "raa": (("bands", "y", "x"), raa_bands.astype("float32")),
        "rho_gc": (("bands", "y", "x"), np.full((nb, ny, nx), 0.1, dtype="float32")),
        "cwav": (("bands",), np.array([443., 865.][:nb], dtype="float32"), {"units": "nm"}),
        "flags": (("y", "x"), flags),
    }, coords={"bands": bands})


@pytest.mark.parametrize("bitmask_invalid", [-1, 1])
def test_rayleigh_per_band_geometry(bitmask_invalid):
    """
    With a per-band viewing geometry, the Rayleigh correction of each band is
    identical to the one obtained with the 2D geometry of this band
    """
    rng = np.random.default_rng(0)
    muv = rng.uniform(0.7, 1.0, (2, 3, 4))
    raa = rng.uniform(0., 180., (2, 3, 4))
    kw = dict(bitmask_invalid=bitmask_invalid)

    block = make_block_rayleigh(muv, raa)
    RayleighCorrection(**kw).process_block(block)
    assert block.rho_rc.dims == ("bands", "y", "x")

    for i, b in enumerate(block.bands.values):
        # same geometry in all bands: the geometry of band i
        block_i = make_block_rayleigh(muv[[i, i]], raa[[i, i]])
        RayleighCorrection(**kw).process_block(block_i)
        for var in ["rho_r", "rho_rg", "t_d", "rho_rc"]:
            np.testing.assert_allclose(
                block[var].sel(bands=b), block_i[var].sel(bands=b), rtol=1e-6)

    if bitmask_invalid > 0:
        assert np.isnan(block.rho_rc[:, 0, 0]).all()
        assert not np.isnan(block.rho_rc[:, 1:, :]).any()


def test_rayleigh_per_band_geometry_differs():
    """
    The per-band geometry is actually used: a different viewing geometry in
    one band changes the Rayleigh reflectance of this band only
    """
    muv = np.full((2, 3, 4), 0.9)
    raa = np.full((2, 3, 4), 90.)
    kw = {}
    ref = make_block_rayleigh(muv, raa)
    RayleighCorrection(**kw).process_block(ref)

    muv2 = muv.copy()
    muv2[1] = 0.7
    block = make_block_rayleigh(muv2, raa)
    RayleighCorrection(**kw).process_block(block)
    np.testing.assert_allclose(block.rho_r.isel(bands=0), ref.rho_r.isel(bands=0))
    assert (np.abs(block.rho_r.isel(bands=1) - ref.rho_r.isel(bands=1)) > 1e-4).all()
