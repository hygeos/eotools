#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from eoread import msi

from eotools.srf import get_SRF, get_SRF_eumetsat, integrate_srf, plot_srf, rename, select, filter_bands, nbands

from . import conftest

level1 = pytest.fixture(msi.get_sample)
  

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
    ],
)
def test_get_srf(request, platform, sensor, sel):
    srf = rename(get_SRF((platform, sensor)), "trim")
    print(srf)
    if sel is not None:
        srf = select(srf, **sel)

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
def test_srf_multidim(platform, sensor):
    ds = xr.Dataset()
    ds.attrs.update(sensor=sensor, platform=platform)
    srf = get_SRF(ds)

    # Merge all bands in a new dimension
    srf = xr.concat([srf[x] for x in srf], dim="band").to_dataset(name="SRF")
    srf = srf.assign_coords(band=["BLUE", "RED", "NIR", "SWIR"])

    # test various integration options
    data = np.linspace(400, 1000, 100)
    x = xr.DataArray(data, dims=['wav'])
    x = x.assign_coords(wav=data)
    x.wav.attrs['units'] = 'nm'

    for kwargs in [
        {"x": lambda x: x},
        {"x": x, "resample": "x"},
        {"x": x, "resample": "srf"},
    ]:
        integrated = integrate_srf(srf, integration_dimension="wavelength", **kwargs)
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
