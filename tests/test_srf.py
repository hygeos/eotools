#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from eoread.reader import msi
from eoread.utils.graphics import plot_srf

from eotools.srf import get_SRF

from . import conftest

level1 = pytest.fixture(msi.get_sample)


@pytest.mark.parametrize(
    "sensor",
    [
        "LANDSAT 8 OLI",
        "GOES 16 ABI",
        "SENTINEL2-A MSI",
    ],
)
def test_get_srf(request, sensor):
    srf = get_SRF(sensor)
    plot_srf(srf)

    conftest.savefig(request, bbox_inches="tight")


def test_srf_from_l1(level1: Path):
    l1 = msi.Level1_MSI(level1)
    srf = get_SRF(l1)
    assert len(srf) == len(l1.bands)
