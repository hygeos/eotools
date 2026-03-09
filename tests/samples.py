#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from core import env

def level1_msi() -> Path:
    msi_prod = (
        env.getdir("DIR_SAMPLES")
        / "SENTINEL-2-MSI"
        / "S2B_MSIL1C_20250320T104639_N0511_R051_T31UDS_20250320T142408.SAFE"
    )
    if not msi_prod.exists():
        raise FileNotFoundError(
            f"MSI sample product not found. Please download {msi_prod.name} in {msi_prod.parent}"
        )
    return msi_prod