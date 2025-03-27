#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from core.conftest import savefig
from eoread import msi
from eoread.common import timeit
from matplotlib import pyplot as plt

from eotools.cm.basic import Cloud_mask

level1 = pytest.fixture(msi.get_sample)


@pytest.mark.parametrize('method', ['apply_ufunc', 'map_blocks'])
def test_cloudmask(level1: Path, method, request):
    ds = msi.Level1_MSI(level1)

    with timeit('Init'):
        Cloud_mask(ds, 'Rtoa', 865, 1).apply(method=method)

    with timeit('Compute'):
        ds = ds.sel(bands=865)[['Rtoa', 'flags']].compute()

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
