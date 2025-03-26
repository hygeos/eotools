#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import pytest
from eoread.ancillary_nasa import Ancillary_NASA
from eoread.eo import init_geometry
from eoread import msi
from eoread.common import timeit
from core.conftest import savefig

from eotools.apply_ancillary import apply_ancillary
from eotools.gaseous_absorption import get_absorption
from eotools.gaseous_correction import Gaseous_correction
from eotools.srf import integrate_srf, rename
from eotools.srf import get_SRF
from core.pytest_utils import parametrize_dict
from scipy.integrate import simpson

level1 = pytest.fixture(msi.get_sample)


@pytest.mark.parametrize('platform,sensor', [
    ('landsat-8', 'oli'),
    ('landsat-9', 'oli'),
])
@pytest.mark.parametrize('gas', ['o3', 'no2'])
@pytest.mark.parametrize('integration_function', **parametrize_dict({
    'simpson': simpson,
    'trapz': np.trapz,
}))
def test_integrate_srf(platform, sensor, gas, integration_function):
    ds = xr.Dataset()
    ds.attrs.update(platform=platform, sensor=sensor)
    srf = get_SRF(ds)
    k = get_absorption(gas)

    for resample in ["x", "srf"]:
        integrated = integrate_srf(
            srf,
            k,
            integration_function=integration_function,
            resample=resample,
        )
        print(resample, integrated)



@pytest.mark.parametrize('method', ['apply_ufunc', 'map_blocks'])
def test_gaseous_correction(level1: Path, method, request):
    ds = msi.Level1_MSI(level1).chunk(bands=-1)
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
    srf = rename(get_SRF(ds), ds.bands.values, thres_check=100)
    with timeit('Init'):
        Gaseous_correction(ds, srf, input_var='Rtoa').apply(method=method)
    plt.plot()
    list_vars = ['Rtoa', 'rho_gc']
    with timeit('Compute'):
        px = ds[list_vars].sel(x=1000, y=1000).compute()
    for varname in list_vars:
        px[varname].plot(label=varname)
    plt.grid(True)
    plt.axis(ymin=0, ymax=0.3)
    plt.legend()
    savefig(request)
    plt.close()
