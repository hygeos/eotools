#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
import xarray as xr
from core.process.blockwise import CompoundProcessor
from core.tests import conftest
from eoread import msi
from eoread.ancillary_nasa import Ancillary_NASA
from eoread.common import timeit
from eoread.eo import init_geometry
from luts import Idx
from matplotlib import pyplot as plt

from eotools.apply_ancillary import ApplyAncillary, apply_ancillary
from eotools.bodhaine import rod
from eotools.geometry import InitGeometry
from eotools.rayleigh import RayleighCorrection
from eotools.rayleigh_legacy import Rayleigh_correction
from eotools.srf import get_SRF, integrate_srf, rename
from tests import samples

level1_msi = pytest.fixture(samples.level1_msi)

def test_calc_odr(level1_msi: Path):
    ds = msi.Level1_MSI(level1_msi, v1_compat=True)
    srf = rename(get_SRF(ds), ds.bands.values, thres_check=100)
    tau_r = integrate_srf(srf, lambda wav_nm: rod(wav_nm * 1e-3))
    print(tau_r)


def test_plot_rho_ray(level1_msi, request):
    ds = msi.Level1_MSI(level1_msi)
    rc = Rayleigh_correction(ds, bitmask_invalid=0)
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
def test_rayleigh_correction(level1_msi: Path, method, request):
    ds = msi.Level1_MSI(level1_msi, v1_compat=True)
    ds = ds.drop(['x', 'y']).unify_chunks()  # TODO shall be removed for v2 compat
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
        Rayleigh_correction(ds, bitmask_invalid=0).apply(method=method)
    with timeit('Compute'):
        px = ds[list_vars].isel(x=1000, y=1000).compute()
    plt.plot()
    for varname in list_vars:
        px[varname].plot(label=varname)
    plt.grid(True)
    plt.legend()
    conftest.savefig(request)



@pytest.mark.parametrize('mode', ['srf', 'wav'])
def test_rayleigh_correction_new(level1_msi: Path, mode: str, request):
    ds = msi.Level1_MSI(level1_msi).isel(x=slice(1000, 1500), y=slice(1000, 1500))
    ds.cwav.attrs.update(units='nm')  # TODO: move in eoread
    ds['altitude'] = xr.zeros_like(ds.latitude, dtype='float32')
    ds['altitude'].attrs.update(units='m')
    ds = ds.chunk(bands=-1)
    srf = get_SRF(ds) if mode == "srf" else None
    compound = CompoundProcessor(
        [
            InitGeometry(ds),
            ApplyAncillary(ds, Ancillary_NASA()),
            RayleighCorrection(srf=srf),
        ]
    )
    ds['rho_gc'] = ds['Rtoa']
    res = compound.map_blocks(ds)
    res = res.compute()

    res.rho_rc.sel(bands='B8A').plot()
    conftest.savefig(request)

    