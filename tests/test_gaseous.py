#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import xarray as xr
from core.tests.conftest import savefig
from core.pytest_utils import parametrize_dict
from eoread import msi
from eoread.ancillary_nasa import Ancillary_NASA
from eoread.common import timeit
from eoread.eo import init_geometry
from matplotlib import pyplot as plt
from scipy.integrate import simpson

from eotools.apply_ancillary import apply_ancillary
from eotools.gaseous_absorption import (
    abs_data_fit,
    gas_list_gatiab,
    get_abs_data,
    get_absorption,
    get_gaseous_abs,
    get_transmission_coeffs,
    trans_func,
    transmission_model,
    transmission_model_eval,
)
from eotools.gaseous_correction import Gaseous_correction

# Check if gatiab is available
try:
    from gatiab import Gatiab  # noqa: F401
    GATIAB_AVAILABLE = True
except ImportError:
    GATIAB_AVAILABLE = False
from eotools.srf import (filter_bands, get_bands, get_SRF, integrate_srf,
                         plot_srf, rename, select, squeeze)

level1 = pytest.fixture(msi.get_sample)


@pytest.mark.parametrize('platform,sensor', [
    ('landsat-8', 'oli'),
    ('landsat-9', 'oli'),
])
@pytest.mark.parametrize('gas', ['o3', 'no2'])
@pytest.mark.parametrize('integration_function', **parametrize_dict({
    'simpson': simpson,
    'trapz': np.trapz,
}))
def test_integrate_srf(platform, sensor, gas, integration_function):
    srf = get_SRF((platform, sensor))
    k = get_absorption(gas)

    for resample in ["x", "srf"]:
        integrated = integrate_srf(
            srf,
            k,
            integration_function=integration_function,
            resample=resample,
        )
        print(resample, integrated)


def get_x_range(
    gas: str, gabs: xr.DataTree, n_air_mass: int = 15, n_gas_content: int = 15
) -> xr.DataArray:
    """Generate a range of normalized optical path values (M * U / U0).

    Creates a 2D grid of air mass and gas content values, then returns
    the outer product flattened to 1D. The gas content range is chosen
    so that U/U0 spans values below and above 1.0, ensuring the fit
    coefficients are well-constrained around the reference column.

    Parameters
    ----------
    gas : str
        Gas species name (e.g. 'H2O', 'O3', 'CO2', 'CH4', 'NO2').
    gabs : xr.DataTree
        High-resolution gaseous absorption model DataTree (from
        :func:`get_gaseous_abs`), providing the nominal gas content
        U0 for the requested species.
    n_air_mass : int, optional
        Number of air mass samples (default 15, range 2.0-6.0).
    n_gas_content : int, optional
        Number of gas content samples (default 15).

    Returns
    -------
    xr.DataArray
        2D array with dimensions ``("air_mass", "gas_content")`` containing
        the product M * (U / U0) for each combination.
    """
    
    air_mass = xr.DataArray(
        np.linspace(2.0, 6.0, n_air_mass),
        dims=["air_mass"],
    )
    U0 = gabs[gas].U0

    if gas == 'H2O':
        U = np.linspace(0.5, 6.5, n_gas_content)
        U_U0 = xr.DataArray(U, dims=["gas_content"])/U0
    elif gas == 'O3':
        U = np.linspace(100.0, 500.0, n_gas_content)
        U_U0 = xr.DataArray(U, dims=["gas_content"])/U0
    else:
        U_U0 = xr.DataArray(
            np.linspace(0.83909181, 1.0266535, n_gas_content),
            dims=["gas_content"],
        )
    assert U_U0[0] < 1
    assert U_U0[-1] > 1

    return air_mass*U_U0

@pytest.mark.parametrize('method', ['map_blocks', 'blockwise'])
@pytest.mark.parametrize('gas_correction', ['o3_legacy', 'ckdmip'])
def test_gaseous_correction(level1: Path, method, gas_correction: str, request):
    """
    This test should be deprecated in favour of test_gaseous_rayleigh
    """
    ds = msi.Level1_MSI(level1, v1_compat=True).chunk(bands=-1)
    ds = ds.drop(['x', 'y'])
    init_geometry(ds)
    apply_ancillary(
        ds,
        Ancillary_NASA(),
        variables={
            "horizontal_wind": "m/s",
            "sea_level_pressure": "hectopascals",
            "total_column_ozone": "Dobson",
            "total_column_water_vapour": "g/cm²",
        },
    )
    ds.attrs.update(platform='Sentinel-2A')
    srf = rename(get_SRF(ds), ds.bands.values, thres_check=100)
    with timeit('Init'):
        Gaseous_correction(
            ds, srf, input_var="Rtoa", gas_correction=gas_correction,
            bands_sel_ckdmip=slice(None),
        ).apply(method=method)
    plt.plot()
    list_vars = ['Rtoa', 'rho_gc']
    with timeit('Compute'):
        px = ds[list_vars].isel(x=1000, y=1000).compute()
    for varname in list_vars:
        px[varname].plot(label=varname)
    plt.grid(True)
    plt.axis(ymin=0, ymax=0.3)
    plt.legend()
    savefig(request)
    plt.close()


@pytest.mark.skipif(not GATIAB_AVAILABLE, reason="gatiab not installed")
@pytest.mark.parametrize(
    "platform_sensor,sel", [
        ("SENTINEL-3A_OLCI", {"ccd_col": 374, "camera": "FM7"}),
        ("SENTINEL-3B_OLCI", {"ccd_col": 374, "camera": "FM7"}),
        ("SENTINEL-2A_MSI", {}),
        ("SENTINEL-2B_MSI", {}),
        ("MSG-1_SEVIRI", {}),
        ("MSG-2_SEVIRI", {}),
        ("MSG-3_SEVIRI", {}),
        ("MSG-4_SEVIRI", {}),
        ("MTG-I1_FCI", {}),
    ]
)
def test_all_gases(platform_sensor: str, sel: Dict, request):
    """
    Generate all coeffs for gatiab supported gases
    """
    coeffs = get_transmission_coeffs(platform_sensor, create=True).sel(sel)
    srf = filter_bands(get_SRF(platform_sensor), 250., 2500.)
    srf = rename(srf, 'trim')
    srf = squeeze(select(srf, **sel))
    cwav = integrate_srf(srf, lambda x: x)
    cwav = np.array([cwav[x].values for x in coeffs.bands.values])

    plot_srf(srf)
    savefig(request)

    print(coeffs)

    # plot the transmissions
    plt.figure()
    plt.yscale("function", functions=(lambda x: x**3, lambda x: x**(1/3)))
    air_mass = 3.
    iwav = np.argsort(cwav)
    tmodel = transmission_model(srf)
    for igas, gas in enumerate(coeffs.gas):
        c = coeffs.sel(gas = gas)
        x = air_mass
        plt.plot(cwav[iwav], np.exp(-c.a * x**c.n)[iwav], f"C{igas}+-", label=str(gas.values))
        plt.plot(
            cwav[iwav],
            transmission_model_eval(tmodel[str(gas.values)].isel(bands=iwav), x),
            f"C{igas}o",
            label=str(gas.values),
        )

    # plot full transmission
    plt.plot(
        cwav[iwav],
        np.exp(-coeffs.a * x**coeffs.n).prod(dim="gas")[iwav],
        "k--",
        label="Total",
    )

    # plot also the other implementation of O3 transmission
    # tau_O3 = get_absorption("o3") * coeffs.U0.sel(gas="O3").values * 1e-3
    # plt.plot(tau_O3['wav'], np.exp(-tau_O3*air_mass), color='gray', ls='--', label='O3 spectrum')

    plt.axis(ymin=0, ymax=1.01)
    plt.yticks([0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
    plt.ylabel('transmission')
    plt.xlabel('wavelength (nm)')
    plt.grid(True)
    plt.legend()
    savefig(request)


@pytest.mark.skipif(not GATIAB_AVAILABLE, reason="gatiab not installed")
@pytest.mark.parametrize('gas', gas_list_gatiab)
@pytest.mark.parametrize("sensor,sel", [
        ("SENTINEL-3A_OLCI", {'ccd_col': 374, 'camera': 'FM7'}),
        ("MSG-1_SEVIRI", {}),
        ("MTG-I1_FCI", {}),
        ("SENTINEL2-A_MSI", {}),
    ])
def test_gaseous_fit(sensor: str, sel: Dict, gas: str, request):

    # load SRF
    srf = get_SRF(sensor)
    srf = filter_bands(srf, 250, 2500)
    srf = rename(srf, 'trim')
    srf = select(srf, **sel)
    cwav = integrate_srf(srf, lambda x: x)
    cwav = xr.concat([cwav[x] for x in cwav], dim='bands')
    
    # load absorption data for fitting
    ds_gas = get_abs_data(srf, gas)

    coeffs = abs_data_fit(ds_gas)
    coeffs1 = abs_data_fit(ds_gas, method="curve_fit")

    list_bands = get_bands(srf)
    for iband in range(len(list_bands)):

        U = ds_gas.U.broadcast_like(ds_gas.trans).isel({"bands": iband}).values.ravel()
        M = ds_gas.M.broadcast_like(ds_gas.trans).isel({"bands": iband}).values.ravel()
        T = ds_gas.trans.isel({"bands": iband}).values.ravel()
        U0 = ds_gas.U0.values

        a = coeffs.a.isel({"bands": iband}).values
        n = coeffs.n.isel({"bands": iband}).values
        # a1 = coeffs1.a.isel({"bands": iband}).values
        # n1 = coeffs1.n.isel({"bands": iband}).values

        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        fig.suptitle(f'{gas}\nband = {list_bands[iband]} ({float(cwav.isel(bands=iband))})')
        
        # Plot regression (log scale)
        X = np.log(M * U / U0)
        ax1.plot(X, np.log(-np.log(T)), 'b+')
        ax1.plot(X, np.log(a) + n * X, 'r--')
        # ax1.plot(X, np.log(a1) + n1 * X, 'y--')
        ax1.set_xlabel('ln(M * U)')
        ax1.set_ylabel(f'ln(-ln(T({gas})))')
        ax1.grid(True)
        ax1.legend()

        # Plot regression (linear scale)
        X_lin = np.linspace(np.amin(M * U / U0), np.amax(M * U / U0), 100)
        ax2.plot(M * U / U0, T, 'b+')
        ax2.plot(X_lin, trans_func(X_lin, a, n), "r--", label=f"a={a:.3g}, n={n:.3g}")
        # ax2.plot(X_lin, trans_func(X_lin, -a1, n1), "y--", label=f"a={-a1:.3g}, n={n1:.3g}")
        ax2.set_xlabel('M * U')
        ax2.set_ylabel(f'T({gas})')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        savefig(request)


@pytest.mark.parametrize("gas", gas_list_gatiab)
def test_get_all_gases(request, gas: str):
    gabs = get_gaseous_abs()
    plt.figure()
    plt.plot(gabs[gas].wav, gabs[gas].T1, label='T1')
    plt.legend()
    plt.grid(True)
    plt.title(f"{gas} transmission")
    savefig(request)


@pytest.mark.parametrize(
    "platform_sensor", [
        "SENTINEL-2A_MSI",
        "MSG-1_SEVIRI",
        "MTG-I1_FCI",
    ])
@pytest.mark.parametrize("gas", gas_list_gatiab)
def test_absorption_model(request, platform_sensor: str, gas: str):
    # Load gaseous absorption model
    gabs = get_gaseous_abs()


    # Load and plot the SRF
    srf = get_SRF(platform_sensor, rename_method='cwav')
    plot_srf(srf)
    savefig(request)

    x_range = get_x_range(gas, gabs, n_air_mass=3, n_gas_content=3)
    x_vals = xr.DataArray(x_range.values.ravel(), dims=('x_vals'))
    T = gabs[gas].T1**x_range
    T_integrated = integrate_srf(srf, T, resample='x')

    # Plot T_integrated for each band as a function of x = M*U/U0
    # => model T = Teq ^ (x^n)
    #    ln(T) = (x^n).ln(Teq)
    #    ln(-ln(T)) = n.ln(x) + ln(-ln(Teq))
    list_bands = get_bands(srf)
    tmodel = transmission_model(srf, x0=5.)
    for iband, band in enumerate(list_bands):
        plt.figure(figsize=(4, 3))
        T_vals = T_integrated[band].values.ravel()

        # Linear scale
        isrt = np.argsort(x_vals)
        plt.plot(x_vals[isrt], T_vals[isrt], 'k+')
        plt.plot(
            x_vals[isrt],
            transmission_model_eval(tmodel[gas].sel(bands=band), x_vals[isrt]),
            "r-"
        )
        plt.xlabel('M · U / U₀')
        plt.ylabel(f'T ({gas})')

        plt.legend()
        plt.title(f'{platform_sensor} - {gas} - {band}')
        plt.grid(True)
        plt.tight_layout()
        savefig(request)



