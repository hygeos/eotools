
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
import xarray as xr
from core.process.blockwise import CompoundProcessor
from core.tests.graphics import xrimshow
from core.tests import conftest
from eoread import msi
from eoread.ancillary_nasa import Ancillary_NASA
from matplotlib import pyplot as plt

from eotools.apply_ancillary import ApplyAncillary
from eotools.dem import InitAltitude
from eotools.gaseous_correction import Gaseous_correction
from eotools.geometry import InitGeometry
from eotools.rayleigh import RayleighCorrection
from eotools.srf import get_SRF
from tests import samples

level1_msi = pytest.fixture(samples.level1_msi)

@pytest.mark.parametrize('mode', ['srf', 'wav'])
def test_gaseous_rayleigh_correction(level1_msi: Path, mode: str, request):
    ds = msi.Level1_MSI(level1_msi, resolution=60).isel(x=slice(600, 780), y=slice(680, 860))
    ds['flags'] = xr.zeros_like(ds.latitude, dtype='uint16')

    ds = ds.chunk(bands=-1)

    if mode == "srf":
        srf = get_SRF(ds, rename_method='bands')
    else:
        srf = None
    compound = CompoundProcessor(
        [
            InitGeometry(ds),
            InitAltitude(),
            ApplyAncillary(ds, Ancillary_NASA()),
            Gaseous_correction(ds, srf, input_var="Rtoa"),
            RayleighCorrection(srf=srf),
        ],
        outputs="all",
    )
    compound.describe()
    ds['rho_gc'] = ds['Rtoa']
    res = compound.map_blocks(ds)
    res = res.compute()

    px = {'x': 660, 'y': 740}

    # Map
    plt.figure()
    _, ax, _ = xrimshow(res.rho_rc.sel(bands='B8A'), yincrease=False)
    ax.plot([px['x']], px['y'], 'ro')
    conftest.savefig(request)

    # Spectrum
    plt.figure(figsize=(4, 3))
    pxdata = res.sel(px)
    plt.plot(ds.cwav, pxdata.Rtoa, label='rho_toa')
    plt.plot(ds.cwav, pxdata.rho_gc, label='rho_gc')
    plt.plot(ds.cwav, pxdata.rho_rc, label='rho_rc')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('reflectance')
    plt.grid(True)
    plt.legend()
    conftest.savefig(request)