import pytest
import tarfile

from eotools.rayleigh_correction import Rayleigh_correction
from lib.eoread.landsat8_oli import Level1_L8_OLI
from tempfile import TemporaryDirectory
from pathlib import Path

@pytest.mark.parametrize('level1',[
    Path('lib/SAMPLE_DATA/LC80140282017275LGN00.tar.gz'),
])
def test_rayleigh_correction(level1):
    with TemporaryDirectory() as tmpdir:
        f = tarfile.open(level1)
        f.extractall(tmpdir) 
        f.close() 
        l1 = Level1_L8_OLI(tmpdir+'/'+level1.name.split('.')[0])
        k = Rayleigh_correction(l1)
        k.apply()