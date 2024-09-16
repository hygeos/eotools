import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from eoread.download_legacy import download_url
from core.fileutils import filegen
from core.tools import only
from core import config


def get_SRF(
    id_sensor: Union[str, tuple, xr.Dataset],
    directory: Optional[Path] = None,
    band_ids: Optional[List[int]] = None,
    check_nbands: bool = True,
    thres_check: Optional[float] = 10,
) -> xr.Dataset:
    """
    Download and store Specrtral Response Function (SRF) from EUMETSAT Website
        -> https://nwp-saf.eumetsat.int/site/software/rttov/download/coefficients/spectral-response-functions/

    Arguments:
        - id_sensor: identifier of the sensor/platform. Can be one of:
            * id_EUMETSAT (str), as in the srf_eumetsat.csv file
            * a tuple (sensor, platform) used for lookup in the sensors.csv
            * an xr.Dataset with sensor and platform attributes
        - directory [Path] : directory path where to save SRF files
        - band_ids [List]: list of sensor band identifiers. If not provided and
          id_sensor is a xr.Dataset, id_sensor.bands is used.
        - check_nbands: whether the number of bands passed as band_ids should match
          the number of files.
        - thres_check: if band id is provided, check that the integrated srf is within
          a distance of `thres_check` of this band_id

    Return a xr.Dataset with the SRF for each band.
    The variable name for each SRF is either a default band identifier
    (integer starting from 1 - if band_ids is None), or corresponds to the
    list `band_ids`.
    """
    # Default directory
    if directory is None:
        directory = config.get("dir_static") / "srf"
    assert directory is not None

    # Load srf_eumetsat.csv
    file_srf_eum = Path(__file__).parent / "srf_eumetsat.csv"
    tar_gz_path = pd.read_csv(file_srf_eum, header=0, index_col=0)

    # Sensor selection
    ds = xr.Dataset()
    if isinstance(id_sensor, str):
        sel = tar_gz_path["id_EUMETSAT"] == id_sensor
        ds.attrs["desc"] = f'Spectral response functions for {id_sensor}'
    else:
        if isinstance(id_sensor, tuple):
            sensor, platform = id_sensor

        elif isinstance(id_sensor, xr.Dataset):
            sensor = id_sensor.sensor
            platform = id_sensor.platform

        else:
            raise TypeError(f"id_sensor has wrong type {id_sensor.__class__}")

        sel = (tar_gz_path["id_sensor"] == sensor.upper()) & (
            tar_gz_path["id_platform"] == platform.upper()
        )
        ds.attrs["desc"] = f'Spectral response functions for {sensor} {platform}'

    urlpath = tar_gz_path[sel]["url_tar"]

    if len(urlpath) != 1:
        raise FileNotFoundError(
            f"No unique corresponding file for {id_sensor}. \
            If {id_sensor} is a new type of data read by eoread module, please update csv file"
        )

    urlpath = urlpath.values[0]

    basename = Path(urlpath).name.split(".")[0]

    # Download and extract the SRF files
    download_extract(directory / basename, urlpath)

    list_files = sorted((directory / basename).glob("*.txt"))

    if ((band_ids is None)
        and isinstance(id_sensor, xr.Dataset)
        and "bands" in id_sensor
        ):
        band_ids = list(id_sensor.bands.values)

    nbands = len(list_files)
    if (band_ids is not None):
        if check_nbands:
            # check that the number of read bands matches `band_ids`
            assert nbands == len(band_ids)
        else:
            nbands = len(band_ids)

    for i, filepath in enumerate(list_files[:nbands]):
        srf = pd.read_csv(
            filepath,
            skiprows=4,
            engine="python",
            dtype=float,
            index_col=False,
            delim_whitespace=True,
            names=["Wavelength", "Response"],
        )
        # Convert wavelength from cm-1 to nm
        srf["Wavelength"] = srf["Wavelength"].apply(lambda x: 1.0 / x * 1e7).values
        with open(filepath) as fp:
            binfo = fp.readline()
        if band_ids is not None:
            bid = band_ids[i]
        else:
            bid = i+1
        ds[bid] = xr.DataArray(
            srf["Response"].values,
            coords={f"wav_{bid}": srf["Wavelength"].values},
            attrs={"band_info": binfo.strip()},
        )
        ds[f"wav_{bid}"].attrs["units"] = "nm"

    if (band_ids is not None) and (thres_check is not None):
        # check that the band id matches the srf
        cwav = integrate_srf(ds, lambda x: x)
        for bid in band_ids:
            diff = abs(float(bid) - cwav[bid])
            assert diff < thres_check, ("There might be an error when providing the "
                f"SRFs. A central wavelength of {cwav[bid]} was found for band {bid}")

    return ds


@filegen(if_exists='skip')
def download_extract(directory: Path, url: str):
    """
    Download a tar.gz file, and extract it to `directory`
    """
    with TemporaryDirectory() as tmpdir:
        tar_gz_path = download_url(url, tmpdir)
        with tarfile.open(tar_gz_path) as f:
            f.extractall(directory)


def integrate_srf(
    srf: xr.Dataset, x: Union[Callable, xr.DataArray],
) -> Dict:
    """
    Integrate the quantity x over each spectral band in srf

    If x is a Callable, it is assumed that it takes inputs as a unit of nm
    If x is a DataArray, it should have a dimension "wav" and an associated unit

    Returns a dict of integrated values
    """
    integrated = {}
    for band, srf_band in srf.items():
        srf_wav = srf_band[only(srf_band.dims)]
        wav = srf_wav.values

        # Calculate quantity x over values `wav`
        # either by interpolation or by calling x
        if callable(x):
            # Check that the input unit is consistent
            assert srf_wav.units == 'nm'
            xx = x(wav)
        elif isinstance(x, xr.DataArray):
            assert x.wav.units == srf_wav.units
            xx = x.interp(wav=srf_wav.values).values
        else:
            raise TypeError(f"Error, x is of type {x.__class__}")

        integrate = np.trapz(srf_band.values * xx, x=wav)
        normalize = np.trapz(srf_band.values, x=wav)
        integrated[band] = integrate / normalize

    return integrated
