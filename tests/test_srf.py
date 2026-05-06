#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from eoread import msi

from eotools.srf import (filter_bands, get_SRF, get_SRF_eumetsat,
                         get_SRF_landsat8_oli, integrate_srf, plot_srf, rename,
                         select)
from core.tests.pytest_utils import parametrize_dict
from tests import samples

from . import conftest

level1 = pytest.fixture(samples.level1_msi)
  

@pytest.mark.parametrize(
    "platform,sensor,sel",
    [
        ("LANDSAT-8", "OLI", None),
        ("sentinel2-A", "MSI", None),
        ("sentinel3-A", "olci", {"camera": "FM7", "ccd_col": 374}),
        ("MSG2", "seviri", None),
        ("ENVISAT", "MERIS", None),
        ("Proba-V", "Proba-V", {"camera": "CENTER"}),
        ("SPOT", "VGT1", None),
        ("SeaHawk", "HawkEye", None),
    ],
)
def test_get_srf(request, platform, sensor, sel):
    srf = rename(get_SRF((platform, sensor)), "trim")
    print(srf)
    if sel is not None:
        srf = select(srf, **sel)

    plot_srf(srf)

    cwav = integrate_srf(srf, lambda x: x)
    print(cwav)

    conftest.savefig(request, bbox_inches="tight")

def test_get_srf2(request):
    srf = rename(get_SRF_landsat8_oli(), "trim")
    print(srf)

    plot_srf(srf)

    conftest.savefig(request, bbox_inches="tight")

@pytest.mark.parametrize('sensor', ["SENTINEL-3A_OLCI"])
def test_select(sensor: str):
    srf = get_SRF(sensor)
    assert 'id' in srf
    sub = select(srf,
                #  band_id=8,
                 camera='FM7',
                 ccd_col=730)
    assert 'id' in sub


@pytest.mark.parametrize('sensor', ['msg_1_seviri', 'sentinel3_1_olci'])
def test_rename_id(sensor: str):
    """
    Test the rename function with sensors containing an "id" variable
    (multidimensional srfs)
    """
    srf = get_SRF_eumetsat(sensor)
    rename(
        srf, [i for i, x in enumerate(srf) if x != "id"], thres_check=None
    )


@pytest.mark.parametrize(
    "platform,sensor",
    [("SPOT", "VGT1")],
)
@pytest.mark.parametrize('srf_multidim', **parametrize_dict({'srf_1dim': False, 'srf_multidim': True}))
@pytest.mark.parametrize('x_kind,resample', **parametrize_dict({
    'x_callable': ('callable', None),
    'x_1dim_srf': ('1dim', 'srf'),
    'x_1dim_x': ('1dim', 'x'),
    'x_multidim_srf': ('multidim', "srf"),
    'x_multidim_x': ('multidim', "x"),
}))
def test_srf_multidim(platform, sensor, srf_multidim: bool, x_kind: str, resample):
    ds = xr.Dataset()
    ds.attrs.update(sensor=sensor, platform=platform)
    srf = get_SRF(ds)

    # Merge all bands in a new dimension
    srf = xr.concat([srf[x] for x in srf], dim="band").to_dataset(name="SRF")
    srf = srf.assign_coords(band=["BLUE", "RED", "NIR", "SWIR"])

    # test various integration options
    if x_kind == 'callable':
        x = lambda y: y  # noqa: E731
    else:
        data = np.linspace(400, 1000, 100)
        x = xr.DataArray(data, dims=['wav'])
        x = x.assign_coords(wav=data)
        if x_kind == 'multidim':
            x = np.stack([x.data] * 3)
            x = xr.DataArray(x, dims=['extra', 'wav'], coords={'extra': [0, 1, 2], 'wav': data})
        x.wav.attrs['units'] = 'nm'
        x.attrs['test_attr'] = 'test_value'

    srf_ = srf if srf_multidim else srf.isel(band=0)

    integrated = integrate_srf(
        srf=srf_,
        x=x,
        integration_dimension="wavelength" if srf_multidim else None,
        resample=resample,
    )
    if isinstance(x, xr.DataArray):
        for band in integrated.data_vars:
            assert integrated[band].attrs.get('test_attr') == 'test_value'
    for dim in integrated.dims:
        assert dim in integrated.coords
    print(integrated)


def test_srf_from_l1(level1: Path):
    l1 = msi.Level1_MSI(level1)
    srf = get_SRF(l1, thres_check=100)
    assert len(srf) == len(l1.bands)


def test_filter_bands():
    """Test the filter_bands function using FCI SRF data"""
    # Get FCI SRF data
    srf = get_SRF_eumetsat('mtg_1_fci')
    
    original_nbands = len([k for k in srf.data_vars if k != "id"])
    
    # Test no filtering
    assert len([k for k in filter_bands(srf).data_vars if k != "id"]) == original_nbands
    
    # Test wavelength range filtering
    srf_filtered = filter_bands(srf, wav_min=250.0, wav_max=2500.0)
    cwav = integrate_srf(srf_filtered, lambda x: x)
    for band in cwav.data_vars:
        assert 250. <= float(cwav[band].values) <= 2500.0
    
    # Test empty result preserves attributes
    srf_empty = filter_bands(srf, wav_min=10000.0, wav_max=20000.0)
    assert srf_empty.attrs == srf.attrs
