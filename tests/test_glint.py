#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from core.tests.conftest import savefig
from eoread import msi
from eoread.ancillary_nasa import Ancillary_NASA
from eoread.common import timeit
from eoread.eo import init_geometry
from matplotlib import pyplot as plt

from eotools.apply_ancillary import apply_ancillary
from eotools.glint import apply_glitter

level1 = pytest.fixture(msi.get_sample)


@pytest.mark.parametrize('method', ['apply_ufunc', 'map_blocks'])
def test_glint(level1: Path, method, request):
    ds = msi.Level1_MSI(level1)
    init_geometry(ds, scat_angle=True)
    apply_ancillary(
        ds,
        Ancillary_NASA(),
        variables={
            "horizontal_wind": "m/s",
            "sea_level_pressure": "hectopascals",
            "total_column_ozone": "Dobson",
        },
    )

    with timeit('Init'):
        apply_glitter(ds, method=method)

    with timeit('Compute'):
        ds = ds.sel(bands=865)[['Rtoa', 'Rgli']].compute()

    for label, data, vmin, vmax in [
        ("Rtoa", ds.Rtoa, 0, 0.1),
        ("Rgli", ds.Rgli, 0, 0.01),
    ]:
        plt.plot()
        plt.imshow(data, vmin=vmin, vmax=vmax)
        plt.title(label)
        plt.colorbar()
        savefig(request)
        plt.close()
