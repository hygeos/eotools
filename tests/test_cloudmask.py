#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from hygeos_core.process.blockwise import CompoundProcessor
from hygeos_core.tests.conftest import savefig
from eoread.flags import FlagsInit
from eoread.readers.msi import Level1C_MSI
from hygeos_core.monitor import Chrono
from matplotlib import pyplot as plt

from eotools.cm.basic import Cloud_mask


@pytest.fixture
def level1() -> Path:
    return Level1C_MSI.get_sample()


def test_cloudmask(level1: Path, request):
    ds = Level1C_MSI(level1, resolution=60)
    ds = ds.chunk(bands=-1)

    with Chrono('Init', unit='s'):
        compound = CompoundProcessor(
            [FlagsInit(
                flags={},
                dtype="uint8",
                flag_reader=ds.attrs["_flag_reader"],
            ),
             Cloud_mask(
                cm_input_var='Rtoa',
                cm_band_nir="B8",
                cm_flag_value=1,
                cm_flag_name='CLOUD',
            )],
            outputs="all",
        )
        ret = compound.map_blocks(ds)

        for var in ret:
            ds[var] = ret[var]

    with Chrono('Compute', unit='s'):
        ds = ds.sel(bands='B8')[['Rtoa', 'flags']].compute()

    for label, data, vmin, vmax in [
        ("Rtoa", ds.Rtoa, 0, 0.1),
        ("CLOUD", ds.flags & 1, 0, 1),
    ]:
        plt.plot()
        plt.imshow(data, vmin=vmin, vmax=vmax)
        plt.title(label)
        plt.colorbar()
        savefig(request)
        plt.close()
