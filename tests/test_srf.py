#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from eoread.utils.graphics import plot_srf

from eotools.srf import get_SRF

from . import conftest


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
