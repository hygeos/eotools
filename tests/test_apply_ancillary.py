from pathlib import Path

import pytest
from eoread.ancillary.nasa import Ancillary_NASA
from eoread.readers.msi import Level1C_MSI
from hygeos_core.tests.pytest_utils import parametrize_dict
from hygeos_core.xrtags import tag_filter
from matplotlib import pyplot as plt
from hygeos_core.tools import datetime

from eotools.apply_ancillary import ApplyAncillary, apply_ancillary

from . import conftest


@pytest.fixture
def level1() -> Path:
    return Level1C_MSI.get_sample()


@pytest.mark.parametrize(
    "ancillary",
    **parametrize_dict({
            "NASA": Ancillary_NASA,
            # "ERA5": ERA5,
    })
)
def test_apply_ancillary(level1: Path, ancillary, request):
    ds = Level1C_MSI(level1, resolution=60).thin(x=10, y=10)
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

@pytest.mark.parametrize(
    "ancillary",
    **parametrize_dict({
            "NASA": Ancillary_NASA,
            # "ERA5": ERA5,
    })
)
def test_apply_ancillary_processor(level1: Path, ancillary, request):
    ds = Level1C_MSI(level1, resolution=60).thin(x=10, y=10).chunk()
    processor = ApplyAncillary(datetime(ds), ancillary())
    # ds = ds.chunk()
    result = processor.map_blocks(ds)
    result = result.compute()

    for x in result:
        plt.figure()
        assert 'units' in result[x].attrs
        result[x].plot() # type: ignore
        conftest.savefig(request)

    