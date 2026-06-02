#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np
import pytest
import xarray as xr
from core.tests.conftest import savefig
from core.pytest_utils import parametrize_dict
from matplotlib import pyplot as plt
from scipy.integrate import simpson

from eotools.gaseous_absorption import (
    gas_list_gatiab,
    get_absorption,
    get_gaseous_abs,
    transmission_model,
    transmission_model_eval,
)
from eotools.srf import (filter_bands, get_band, get_bands, get_SRF, integrate_srf,
                         plot_srf, rename, select, squeeze)

# Check if gatiab is available
try:
    from gatiab import Gatiab  # noqa: F401
    GATIAB_AVAILABLE = True
except ImportError:
    GATIAB_AVAILABLE = False


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
    srf = filter_bands(get_SRF(platform_sensor), 250., 2500.)
    srf = rename(srf, 'trim')
    srf = squeeze(select(srf, **sel))
    cwav = integrate_srf(srf, lambda x: x)
    cwav = np.array([cwav[x].values for x in cwav])

    plot_srf(srf)
    savefig(request)

    # plot the transmissions
    plt.figure()
    plt.yscale("function", functions=(lambda x: x**3, lambda x: x**(1/3)))
    air_mass = xr.DataArray(3.)
    iwav = np.argsort(cwav)
    tmodel = transmission_model(srf)
    Ttot = np.array(1.)
    for gas in gas_list_gatiab:
        x = air_mass
        T = transmission_model_eval(tmodel[gas].ds.isel(bands=iwav), x)
        plt.plot(
            cwav[iwav],
            T,
            "o-",
            label=gas,
        )
        if Ttot is None:
            Ttot = T
        else:
            Ttot = Ttot * T

    # plot full transmission
    plt.plot(
        cwav[iwav],
        Ttot,
        "k--",
        label="Total",
    )

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
    
    # Load high-resolution gaseous absorption model
    gabs = get_gaseous_abs()

    # Build x = M * U / U0 grid and compute integrated transmission
    x_range = get_x_range(gas, gabs)
    T = gabs[gas].T1 ** x_range
    T_integrated = integrate_srf(srf, T, resample='x')

    tmodel = transmission_model(srf)

    list_bands = get_bands(srf)
    for iband in range(len(list_bands)):

        x_vals = x_range.values.ravel()
        T = get_band(T_integrated, list_bands[iband]).values.ravel()

        Teq = tmodel[gas].Teq.isel({"bands": iband}).values
        n = tmodel[gas].n.isel({"bands": iband}).values
        a = -np.log(Teq)

        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        fig.suptitle(
            f"{gas}\nband = {list_bands[iband]} ({float(get_band(cwav, list_bands[iband]))})"
        )
        
        # Plot regression (log scale)
        X = np.log(x_vals)
        ax1.plot(X, np.log(-np.log(T)), 'b+')
        ax1.plot(X, np.log(a) + n * X, 'r--')
        ax1.set_xlabel('ln(M · U / U₀)')
        ax1.set_ylabel(f'ln(-ln(T({gas})))')
        ax1.grid(True)
        ax1.legend()

        # Plot regression (linear scale)
        X_lin = np.linspace(np.amin(x_vals), np.amax(x_vals), 100)
        ax2.plot(x_vals, T, 'b+')
        ax2.plot(X_lin, np.exp( -a*X_lin**n ), "r--", label=f"a={a:.3g}, n={n:.3g}")
        ax2.set_xlabel('M · U / U₀')
        ax2.set_ylabel(f'T({gas})')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        savefig(request)

def plot_all_gases(wav_min=380, wav_max=1100):
    gabs = get_gaseous_abs()
    wav = gabs["CH4"].wav
    mask = (wav_min <= wav) & (wav <= wav_max)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot H2O in the background
    for gas, color in [
        ("H2O", "lightgray"),
        ("CH4", None),
        ("CO2", None),
        ("N2", None),
        ("N2O", None),
        ("O2", None),
        ("O3", None),
    ]:
        kwargs = {"label": gas}
        if color is not None:
            kwargs["color"] = color
        ax.plot(gabs[gas].wav[mask], gabs[gas].T1[mask], **kwargs)

    ax.set_yscale("function", functions=(lambda x: x**3, lambda x: x**(1/3)))
    ax.set_ylim(0, 1.01)
    ax.set_yticks([0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Transmission $T_1(\lambda)$")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def test_plot_all_gases(request):
    plot_all_gases()
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



