#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pytest
from eoread.ancillary_nasa import Ancillary_NASA
from eoread.eo import init_geometry
from eoread.reader import msi

from eotools.apply_ancillary import apply_ancillary
from eotools.gaseous_absorption import get_absorption
from eotools.gaseous_correction import Gaseous_correction
from eotools.srf import integrate_srf
from eotools.srf import get_SRF
from core.pytest_utils import parametrize_dict
from scipy.integrate import simpson

level1 = pytest.fixture(msi.get_sample)


@pytest.mark.parametrize('sensor', [
    'landsat_8_oli',
    'landsat_9_oli',
])
@pytest.mark.parametrize('gas', ['o3', 'no2'])
@pytest.mark.parametrize('integration_function', **parametrize_dict({
    'simpson': simpson,
    'trapz': np.trapz,
}))
def test_integrate_srf(sensor, gas, integration_function):
    srf = get_SRF(sensor)
    k = get_absorption(gas)

    integrated = integrate_srf(srf, k, integration_function=integration_function)

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
    srf = get_SRF(ds, thres_check=100)
    Gaseous_correction(ds, srf, input_var='Rtoa').apply()
