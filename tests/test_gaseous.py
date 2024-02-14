#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from eoread.ancillary_nasa import Ancillary_NASA
from eoread.eo import init_geometry
from eoread.reader import msi
from eoread.utils.tools import datetime

from eotools.apply_ancillary import apply_ancillary
from eotools.gaseous_absorption import get_absorption
from eotools.gaseous_correction import Gaseous_correction
from eotools.srf import integrate_srf
from eotools.srf import get_SRF

level1 = pytest.fixture(msi.get_sample)


@pytest.mark.parametrize('sensor, platform',[
    ('OLI', 'Landsat8'),
    ('OLI', 'Landsat9'),
])
@pytest.mark.parametrize('gas', ['o3', 'no2'])
def test_integrate_srf(sensor, platform, gas):
    srf = get_SRF((sensor, platform))
    k = get_absorption(gas)

    integrated = integrate_srf(srf, k)

    print(integrated)


def test_gaseous_correction(level1: Path):
    ds = msi.Level1_MSI(level1)
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
    srf = get_SRF(ds)
    Gaseous_correction(ds, srf).apply()
