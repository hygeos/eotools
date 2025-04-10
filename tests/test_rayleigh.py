#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from matplotlib import pyplot as plt
import pytest
import xarray as xr
from eoread.ancillary_nasa import Ancillary_NASA
from eoread.eo import init_geometry
from eoread import msi
from luts import Idx
from eoread.common import timeit
from core.conftest import savefig

from eotools.apply_ancillary import apply_ancillary
from eotools.bodhaine import rod
from eotools.rayleigh_legacy import Rayleigh_correction
from eotools.srf import get_SRF, integrate_srf, rename

from . import conftest

level1 = pytest.fixture(msi.get_sample)


def test_calc_odr(level1: Path):
    ds = msi.Level1_MSI(level1)
    srf = rename(get_SRF(ds), ds.bands.values, thres_check=100)
    tau_r = integrate_srf(srf, lambda wav_nm: rod(wav_nm * 1e-3))
    print(tau_r)


def test_plot_rho_ray(level1, request):
    ds = msi.Level1_MSI(level1)
    rc = Rayleigh_correction(ds)
    rc.mlut.describe()
    # [0] Rmolgli (float32 in [0, 6.5e+04]), axes=('dim_mu', 'dim_phi', 'dim_mu', 'dim_tauray', 'dim_wind')
    # [1] Rmol (float32 in [0, 6.53]), axes=('dim_mu', 'dim_phi', 'dim_mu', 'dim_tauray')
    # [2] Tmolgli (float32 in [0.366, 1]), axes=('dim_mu', 'dim_tauray', 'dim_wind')
    sub = rc.mlut["Rmolgli"].sub()[:, :, Idx(0.5), Idx(0.2), Idx(5.0)]
    sub.to_xarray().plot()  # type: ignore
    conftest.savefig(request)

    # Check glint position
    assert sub[Idx(0.5), Idx(180)] > sub[Idx(0.3), Idx(0)]  # type: ignore


@pytest.mark.parametrize('method', ['apply_ufunc', 'map_blocks'])
def test_rayleigh_correction(level1: Path, method, request):
    ds = msi.Level1_MSI(level1)
    ds = ds.chunk(bands=-1)
    init_geometry(ds)
    apply_ancillary(
        ds,
        Ancillary_NASA(),
        variables={
            "horizontal_wind": "m/s",
            "sea_level_pressure": "hectopascals",
            "total_column_ozone": "Dobson",
        },
    )
    ds["altitude"] = xr.zeros_like(ds["total_column_ozone"])

    ds["rho_gc"] = ds.Rtoa

    list_vars = ['Rtoa', 'Rprime', 'rho_r']
    with timeit('Init'):
        Rayleigh_correction(ds).apply(method=method)
    with timeit('Compute'):
        px = ds[list_vars].sel(x=1000, y=1000).compute()
    plt.plot()
    for varname in list_vars:
        px[varname].plot(label=varname)
    plt.grid(True)
    plt.legend()
    savefig(request)

