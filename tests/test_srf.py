#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from tempfile import TemporaryDirectory
from eotools.srf import get_SRF, combine_with_srf, get_absorption

@pytest.mark.parametrize('sensor, platform',[
    ('OLI', 'Landsat8'),
    ('OLI', 'Landsat9'),
])
def test_srf_compute(sensor, platform):
    get_SRF(sensor, platform)

@pytest.mark.parametrize('sensor, platform',[
    ('OLI', 'Landsat8'),
    ('OLI', 'Landsat9'),
])
def test_combine_srf(sensor, platform):
    with TemporaryDirectory() as tmpdir:
        srf = get_SRF(sensor, platform)
        k = get_absorption('o3', tmpdir)
        combine_with_srf(srf, k)
    