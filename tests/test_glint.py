#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from hygeos_core.tests.conftest import savefig
from eoread.ancillary.nasa import Ancillary_NASA
from eoread.readers.msi import Level1C_MSI
from hygeos_core.monitor import Chrono
from matplotlib import pyplot as plt

from eotools.apply_ancillary import ApplyAncillary
from eotools.geometry import InitGeometry
from eotools.glint import CalcSunGlint
from hygeos_core.process.blockwise import CompoundProcessor
from hygeos_core.tools import datetime


@pytest.fixture
def level1() -> Path:
    return Level1C_MSI.get_sample()

# TODO: check sza and vza for MSI sample

def test_glint(level1: Path, request):
    ds = Level1C_MSI(level1, resolution=60)
    ret = CompoundProcessor(
        [
            InitGeometry(ds, calc_scat_angle=True),
            ApplyAncillary(datetime(ds), Ancillary_NASA()),
            CalcSunGlint(),
        ],
        outputs="all",
    ).map_blocks(ds)

    for var in ret:
        ds[var] = ret[var]

    with Chrono('Compute', unit='s'):
        ds = ds.sel(bands='B8').compute()

    for label, data, vmin, vmax in [
        ("sza", ds.sza, 0, None), # DEBUG
        ("vza", ds.vza, 0, None), # DEBUG
        ("mus", ds.mus, 0, None), # DEBUG
        ("muv", ds.muv, 0, None), # DEBUG
        ("horizontal_wind", ds.horizontal_wind, 0, None),
        ("Rtoa", ds.Rtoa, 0, None),
        ("Rgli", ds.Rgli, 0, None),
    ]:
        plt.plot()
        plt.imshow(data, vmin=vmin, vmax=vmax)
        plt.title(f'{label} = {data[0,0].values}')
        plt.colorbar()
        savefig(request)
        plt.close()
