#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import xarray as xr
from eoread.reader import msi

from eotools.srf import get_SRF, plot_srf, rename, select

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

