#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from eoread.reader import msi

from eotools.srf import get_SRF, plot_srf, select

from . import conftest

level1 = pytest.fixture(msi.get_sample)
  
msi_bands = [
    443 , 490, 560 , 665, 705 , 740,
    783 , 842, 865 , 945, 1375, 1610,
    2190]

@pytest.mark.parametrize(
    "sensor,kw,sel",
    [
        ("landsat_8_oli", {}, None),
        ("sentinel2_1_msi", {"band_ids": msi_bands, "thres_check": 100}, None),
        ("sentinel3_1_olci", {}, {"camera": "FM7", "ccd_col": 374}),
        (("MSG2", "seviri"), {}, None),
        (("ENVISAT", "MERIS"), {}, None),
        ("Proba-V", {}, {"camera": "CENTER"}),
        ("VGT1", {}, None),
    ],
)
def test_get_srf(request, sensor, kw, sel):
    srf = get_SRF(sensor, **kw)
    if sel is not None:
        srf = select(srf, **sel)

    plot_srf(srf)

    conftest.savefig(request, bbox_inches="tight")


def test_srf_from_l1(level1: Path):
    l1 = msi.Level1_MSI(level1)
    srf = get_SRF(l1, thres_check=100)
    assert len(srf) == len(l1.bands)
