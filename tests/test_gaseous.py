#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tarfile

from eotools.gaseous_correction import Gaseous_correction
from eoread.landsat8_oli import Level1_L8_OLI
from tempfile import TemporaryDirectory
from pathlib import Path

@pytest.mark.parametrize('level1',[
    Path('lib/SAMPLE_DATA/LC80140282017275LGN00.tar.gz'),
])
def test_gaseous_correction(level1):
    with TemporaryDirectory() as tmpdir:
        f = tarfile.open(level1)
        f.extractall(tmpdir) 
        f.close() 
        l1 = Level1_L8_OLI(tmpdir+'/'+level1.name.split('.')[0])
        k = Gaseous_correction(l1).apply().compute()