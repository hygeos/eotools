#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from eoread.reader import msi
from eoread.utils.graphics import plot_srf

from eotools.srf import get_SRF

from . import conftest

level1 = pytest.fixture(msi.get_sample)
  
msi_bands = [
    443 , 490, 560 , 665, 705 , 740,
    783 , 842, 865 , 945, 1375, 1610,
    2190]

@pytest.mark.parametrize(
    "sensor,bands",
    [
        ("LANDSAT 8 OLI", None),
        ("GOES 16 ABI", None),
        ("SENTINEL2-A MSI", msi_bands),
    ],
)
def test_get_srf(request, sensor, bands):
    srf = get_SRF(sensor, band_ids=bands, thres_check=100)
    plot_srf(srf)

    conftest.savefig(request, bbox_inches="tight")


def test_srf_from_l1(level1: Path):
    l1 = msi.Level1_MSI(level1)
    srf = get_SRF(l1, thres_check=100)
    assert len(srf) == len(l1.bands)
