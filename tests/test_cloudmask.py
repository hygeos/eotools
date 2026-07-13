#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from core.process.blockwise import CompoundProcessor
from core.tests.conftest import savefig
from eoread import msi
from eoread.common import timeit
from eoread.flags import FlagsInit
from matplotlib import pyplot as plt

from eotools.cm.basic import Cloud_mask

level1 = pytest.fixture(msi.get_sample)


def test_cloudmask(level1: Path, request):
    ds = msi.Level1_MSI(level1)
    ds = ds.chunk(bands=-1)

    with timeit('Init'):
        compound = CompoundProcessor(
            [FlagsInit(
                flags={},
                dtype="uint8",
                flag_reader="eoread.msi.FlagsReader_MSI",
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

    with timeit('Compute'):
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
