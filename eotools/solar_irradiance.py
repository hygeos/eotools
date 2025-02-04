from typing import Literal
import xarray as xr
from core.download import download_url
from core.env import getdir


def solar_irradiance(source: str, **kwargs) -> xr.DataArray:
    """
    Get solar irradiance from various sources (xarray DataArray)

    Ex: solar_irradiance('LISIRD', '1nm')
    """
    if source == "LISIRD":
        return solar_irradiance_lisird(**kwargs).SSI
    else:
        raise ValueError


def solar_irradiance_lisird(
    variant: Literal["full", "p005nm", "p025nm", "p1nm", "1nm"],
) -> xr.Dataset:
    """
    Get solar irradiance from LASP LISIRD

    https://lasp.colorado.edu/lisird/data/tsis1_hsrs_p1nm

    The data file is written in the directory $DIR_STATIC (see core.env.getdir)

    Args:
        variant: variant of the model

    Returns a dataset with variables:
        'SSI': Spectral Solar Irradiance
        'SSI_UNC': Spectral Solar Irradiance Uncertainty
    """
    filename = {
        "full": "hybrid_reference_spectrum_c2022-11-30_with_unc.nc",
        "p005nm": "hybrid_reference_spectrum_p005nm_resolution_c2022-11-30_with_unc.nc",
        "p025nm": "hybrid_reference_spectrum_p025nm_resolution_c2022-11-30_with_unc.nc",
        "p1nm": "hybrid_reference_spectrum_p1nm_resolution_c2022-11-30_with_unc.nc",
        "1nm": "hybrid_reference_spectrum_1nm_resolution_c2022-11-30_with_unc.nc",
    }[variant]
    url_base = 'https://lasp.colorado.edu/lisird/resources/lasp/hsrs/v2/'
    dirname = getdir("DIR_STATIC") / "solar_irradiance"
    filepath = download_url(url_base + filename, dirname=dirname)
    data = xr.open_dataset(filepath)

    return data.assign_coords(wavelength=data['Vacuum Wavelength'])

