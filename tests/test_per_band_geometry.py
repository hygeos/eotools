"""
Per-band viewing geometry: the view angles (vza, vaa) and the derived variables
have a `bands` dimension. For each band, the results shall be identical to the
ones obtained with the 2D geometry of this band.
"""
import numpy as np
import pytest
import xarray as xr

from eotools.gaseous_correction import Gas_correction_CKDMIP, Gas_correction_O3
from eotools.geometry import InitGeometry, reduce_geometry
from eotools.glint import CalcSunGlint
from eotools.srf import get_SRF_gaussian

BANDS = [443, 560, 865]
SHP = (6, 8)


def make_l1(vza: np.ndarray, vaa: np.ndarray) -> xr.Dataset:
    """
    Dataset with 2D sun angles, and view angles `vza`, `vaa` which are either
    2D (y, x) or 3D (bands, y, x)
    """
    rng = np.random.default_rng(1)
    vdims = ("bands", "y", "x") if vza.ndim == 3 else ("y", "x")
    return xr.Dataset({
        "sza": (("y", "x"), rng.uniform(20, 60, SHP).astype("float32")),
        "saa": (("y", "x"), rng.uniform(100, 200, SHP).astype("float32")),
        "vza": (vdims, vza.astype("float32")),
        "vaa": (vdims, vaa.astype("float32")),
        "horizontal_wind": (("y", "x"), np.full(SHP, 5., dtype="float32"), {"units": "m/s"}),
        "latitude": (("y", "x"), np.full(SHP, 45., dtype="float32")),
        "longitude": (("y", "x"), np.full(SHP, 5., dtype="float32")),
        "total_column_ozone": (("y", "x"), np.full(SHP, 300., dtype="float32"), {"units": "Dobson"}),
        "total_column_water_vapour": (("y", "x"), np.full(SHP, 2., dtype="float32"), {"units": "g/cm²"}),
        "sea_level_pressure": (("y", "x"), np.full(SHP, 1013., dtype="float32"), {"units": "hPa"}),
        "rho_gc": (("bands", "y", "x"), np.full((len(BANDS),) + SHP, 0.1, dtype="float32")),
        "cwav": (("bands",), np.array(BANDS, dtype="float32"), {"units": "nm"}),
    }, coords={"bands": BANDS}).chunk(y=3, x=4)


@pytest.fixture
def view_angles():
    rng = np.random.default_rng(0)
    vza = rng.uniform(0, 12, (len(BANDS),) + SHP)
    vaa = rng.uniform(0, 360, (len(BANDS),) + SHP)
    return vza, vaa


def per_band_and_2d(proc_factory, view_angles, var, prepare=None):
    """
    Run a processor with the per-band geometry, and with the 2D geometry of
    each band. Returns (result per band, [result 2D of band i])
    """
    vza, vaa = view_angles

    def run(ds):
        ds = ds.merge(InitGeometry(ds, calc_air_mass=True, calc_scat_angle=True).map_blocks(ds))
        if prepare is not None:
            ds = prepare(ds)
        return proc_factory(ds).map_blocks(ds)[var].compute()

    res = run(make_l1(vza, vaa))
    res_2d = [run(make_l1(vza[i], vaa[i])) for i in range(len(BANDS))]
    return res, res_2d


def test_reduce_geometry():
    da2 = xr.DataArray(np.zeros(SHP), dims=("y", "x"))
    assert reduce_geometry(da2) is da2
    da3 = xr.DataArray(np.arange(3)[:, None, None] * np.ones(SHP),
                       dims=("bands", "y", "x"), coords={"bands": BANDS})
    assert reduce_geometry(da3).dims == ("y", "x")
    assert (reduce_geometry(da3) == 0).all()
    assert (reduce_geometry(da3, band=865) == 2).all()


@pytest.mark.parametrize("var", ["mus", "muv", "raa", "air_mass", "scat_angle"])
def test_init_geometry(view_angles, var):
    vza, vaa = view_angles
    ds3 = make_l1(vza, vaa)
    res = InitGeometry(ds3, calc_air_mass=True, calc_scat_angle=True).map_blocks(ds3)[var].compute()
    if var == "mus":
        assert "bands" not in res.dims
    else:
        assert res.dims == ("bands", "y", "x")  # same as vza
    for i, b in enumerate(BANDS):
        ds2 = make_l1(vza[i], vaa[i])
        ref = InitGeometry(ds2, calc_air_mass=True, calc_scat_angle=True).map_blocks(ds2)[var].compute()
        assert "bands" not in ref.dims
        np.testing.assert_allclose(
            reduce_geometry(res, band=b).transpose("y", "x"), ref, rtol=1e-6)


def test_sun_glint(view_angles):
    res, res_2d = per_band_and_2d(lambda ds: CalcSunGlint(), view_angles, "Rgli")
    assert res.dims == ("bands", "y", "x")
    for i, b in enumerate(BANDS):
        assert res_2d[i].dims == ("y", "x")
        np.testing.assert_allclose(res.sel(bands=b), res_2d[i], rtol=1e-6)


def test_gaseous_ckdmip(view_angles):
    srf = get_SRF_gaussian(
        xr.DataArray(BANDS, dims=["bands"], coords={"bands": BANDS}),
        xr.DataArray([20., 20., 40.], dims=["bands"], coords={"bands": BANDS}),
    )
    res, res_2d = per_band_and_2d(
        lambda ds: Gas_correction_CKDMIP(srf), view_angles, "rho_gc")
    for i, b in enumerate(BANDS):
        np.testing.assert_allclose(res.sel(bands=b), res_2d[i].sel(bands=b), rtol=1e-6)
    # the per-band air mass is actually used
    assert not np.allclose(res.sel(bands=443), res_2d[1].sel(bands=443))


def test_gaseous_o3(view_angles):
    K_OZ = {443: 3e-3, 560: 1e-1, 865: 4e-3}
    res, res_2d = per_band_and_2d(
        lambda ds: Gas_correction_O3(ds, None, None, K_OZ=K_OZ), view_angles, "rho_gc")
    for i, b in enumerate(BANDS):
        np.testing.assert_allclose(res.sel(bands=b), res_2d[i].sel(bands=b), rtol=1e-6)
    assert not np.allclose(res.sel(bands=560), res_2d[0].sel(bands=560))
