#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from eoread import msi

from eotools.srf import get_SRF, integrate_srf, plot_srf, rename, select

from . import conftest

level1 = pytest.fixture(msi.get_sample)
  
msi_bands = [
    443 , 490, 560 , 665, 705 , 740,
    783 , 842, 865 , 945, 1375, 1610,
    2190]

@pytest.mark.parametrize(
    "platform,sensor,ren_kw,sel",
    [
        ("LANDSAT-8", "OLI", {}, None),
        ("sentinel2-A", "MSI", {"band_ids": msi_bands, "thres_check": 100}, None),
        ("sentinel3_A", "olci", {}, {"camera": "FM7", "ccd_col": 374}),
        ("MSG2", "seviri", {}, None),
        ("ENVISAT", "MERIS", {}, None),
        ("Proba-V", "Proba-V", {}, {"camera": "CENTER"}),
        ("SPOT", "VGT1", {}, None),
    ],
)
def test_get_srf(request, platform, sensor, ren_kw, sel):
    ds = xr.Dataset()
    ds.attrs.update(sensor=sensor, platform=platform)
    srf = get_SRF(ds)
    if len(ren_kw) > 0:
        srf = rename(srf, **ren_kw)
    if sel is not None:
        srf = select(srf, **sel)

    plot_srf(srf)

    conftest.savefig(request, bbox_inches="tight")


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
