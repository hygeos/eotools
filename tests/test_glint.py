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

from eotools.apply_ancillary import apply_ancillary, ApplyAncillary
from eotools.glint import apply_glitter, CalcSunGlint
from core.process.blockwise import CompoundProcessor
from core.tools import datetime

level1 = pytest.fixture(msi.get_sample)

# TODO: check sza and vza for MSI sample

def test_glint(level1: Path, request):
    ds = msi.Level1_MSI(level1, v1_compat=True)
    init_geometry(ds, scat_angle=True)
    ret = CompoundProcessor(
        [ApplyAncillary(datetime(ds), Ancillary_NASA()), CalcSunGlint()]
    ).map_blocks(ds)

    for var in ret:
        ds[var] = ret[var]

    with timeit('Compute'):
        ds = ds.sel(bands=865).compute()

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
