#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tarfile

from tempfile import TemporaryDirectory
from eotools.srf import get_SRF, combine_with_srf, get_absorption
from eotools.odr import get_ODR_for_sensor
from eoread.landsat8_oli import Level1_L8_OLI
from pathlib import Path


@pytest.mark.parametrize('level1',[
    Path('lib/SAMPLE_DATA/LC80140282017275LGN00.tar.gz'),
])
def test_srf_from_l1(level1):
    with TemporaryDirectory() as tmpdir:
        f = tarfile.open(level1)
        f.extractall(tmpdir) 
        f.close() 
        l1 = Level1_L8_OLI(tmpdir+'/'+level1.name.split('.')[0])
        srf = get_SRF(l1_ds=l1)

        assert all(b in list(l1.bands.values) for b in list(srf.keys()))
        
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

@pytest.mark.parametrize('sensor, platform',[
    ('OLI', 'Landsat8'),
    ('OLI', 'Landsat9'),
])
def test_odr_compute(sensor, platform):
    srf = get_SRF(sensor, platform)
    get_ODR_for_sensor(srf)