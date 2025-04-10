from pathlib import Path

import pytest
from eoread.ancillary_nasa import Ancillary_NASA
from eoread import msi
from core.pytest_utils import parametrize_dict
from core.xrtags import tag_filter
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
    apply_ancillary(
        ds,
        ancillary(),
        variables={
            "horizontal_wind": "m/s",
            "sea_level_pressure": "hectopascals",
            "total_column_ozone": "Dobson",
        },
    )

    for x in tag_filter(ds, "ancillary"):
        plt.figure()
        ds[x].plot() # type: ignore
        conftest.savefig(request)
