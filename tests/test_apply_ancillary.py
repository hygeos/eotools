from pathlib import Path

import pytest
from eoread.ancillary_nasa import Ancillary_NASA
from eoread.reader import msi
from eoread.utils.pytest_utils import parametrize_dict
from eoread.utils.tools import datetime
from eoread.utils.xrtags import tag_filter
from matplotlib import pyplot as plt

from eotools.apply_ancillary import apply_ancillary

from . import conftest

level1 = pytest.fixture(msi.get_sample)


@pytest.mark.parametrize(
    "ancillary",
    **parametrize_dict({
            "NASA": Ancillary_NASA,
            # "ERA5": ERA5,
    })
)
def test_apply_ancillary(level1: Path, ancillary, request):
    ds = msi.Level1_MSI(level1).thin(x=10, y=10)
    anc = ancillary().get(datetime(ds))
    apply_ancillary(ds, anc)
    
    for x in tag_filter(ds, "ancillary"):
        plt.figure()
        ds[x].plot() # type: ignore
        conftest.savefig(request)
