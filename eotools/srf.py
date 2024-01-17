import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import pandas as pd
import xarray as xr
from eoread.download_legacy import download_url
from eoread.utils.config import load_config
from eoread.utils.fileutils import filegen


def get_SRF(
    id_sensor: Union[str, tuple, xr.Dataset],
    directory: Optional[Path] = None,
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

    Return a xr.Dataset with the SRF for each band.
    """
    # Default directory
    if directory is None:
        directory = load_config()['dir_static'] / "srf"
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

        sel = (tar_gz_path["id_sensor"] == sensor) & (
            tar_gz_path["id_platform"] == platform
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

    for filepath in (directory / basename).glob("*.txt"):
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
            bid = int(binfo.split(",")[0].strip())
        ds[bid] = xr.DataArray(
            srf["Response"].values,
            coords={f"wav_{bid}": srf["Wavelength"].values},
            attrs={"band_info": binfo.strip()},
        )
        ds[f"wav_{bid}"].attrs["units"] = "nm"

    return ds


@filegen()
def download_extract(directory: Path, url: str):
    """
    Download a tar.gz file, and extract it to `directory`
    """
    with TemporaryDirectory() as tmpdir:
        tar_gz_path = download_url(url, tmpdir)
        with tarfile.open(tar_gz_path) as f:
            f.extractall(directory)
