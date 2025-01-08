import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Union
from warnings import warn
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from core.fileutils import filegen
from core.tools import only
from core import env
from core.download import download_url

from scipy.integrate import simpson
from eotools.bodhaine import rod

def get_SRF(
    id_sensor: str | tuple | xr.Dataset,
    **kwargs
) -> xr.Dataset:
    """
    Get a SRF from multiple sources

    Args:
        id_sensor: can be one of:
            - str: identifies the sensor/platform as in the EUMETSAT database
              ex: "msg_1_seviri"
            - tuple of (platform, sensor) or simply a str
              identifies platform/sensors case by case 
              ex: ("Sentinel3-A", "OLCI")
            - a dataset, which contains platform and sensor attributes

        Other **kwargs are passed to get_SRF_eumetsat, check its doc for more
        information.

    Return a xr.Dataset with the SRF for each band.
    The variable name for each SRF is either a default band identifier
    (integer starting from 1 - if band_ids is None), or corresponds to the
    list `band_ids`.
    """
    if isinstance(id_sensor, xr.Dataset):
        id_sensor = (id_sensor.platform, id_sensor.sensor)

    if id_sensor == ("LANDSAT8", "OLI"):
        srf = get_SRF_landsat8_oli()
    elif id_sensor in ["VGT1", "VGT2"]:
        srf = get_SRF_vgt(id_sensor)
    elif id_sensor == "MISR":
        srf = get_SRF_misr()
    elif id_sensor == "Proba-V":
        srf = get_SRF_probav()
    elif id_sensor[1] == "OLCI": # ("S3X", "OLCI")
        srf = get_SRF_olci(id_sensor[0])
    elif id_sensor == ("ENVISAT", "MERIS"):
        srf = get_SRF_meris()
    elif id_sensor[1] == "MSI":  # ("S2X", "MSI")
        srf = get_SRF_msi(id_sensor[0])
    else:
        # map other platform/sensor to EUMETSAT identifier
        id_sensor = {
            ("MSG1", "seviri"): "msg_1_seviri",
            ("MSG2", "seviri"): "msg_2_seviri",
            ("MSG3", "seviri"): "msg_3_seviri",
            ("MSG4", "seviri"): "msg_4_seviri",
        }.get(id_sensor, id_sensor) # type: ignore

        assert isinstance(id_sensor, str)
        srf = get_SRF_eumetsat(id_sensor, **kwargs)

    if "desc" not in srf.attrs:
        srf.attrs["desc"] = f'Spectral response functions for {id_sensor}'
    if "sensor" not in srf.attrs:
        srf.attrs["sensor"] = id_sensor

    return srf


def get_SRF_eumetsat(
    id_sensor: str,
    band_ids: Optional[List[int]] = None,
    check_nbands: bool = True,
    thres_check: Optional[float] = 10,
) -> xr.Dataset:
    """
    Download and store Specrtral Response Function (SRF) from EUMETSAT Website
        -> https://nwp-saf.eumetsat.int/site/software/rttov/download/coefficients/spectral-response-functions/

    Arguments:
        - id_sensor: identifier of the sensor/platform (as in the srf_eumetsat.csv file)
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
    
    empty_link = "https://nwp-saf.eumetsat.int/downloads/rtcoef_rttov13/ir_srf/rtcoef_{}_srf/rtcoef_{}_srf.tar.gz"

    # Default directory
    directory = env.getdir("DIR_STATIC") / "srf"

    # Load srf_eumetsat.csv to check the list of available sensors
    # and raise a warning if needed
    file_srf_eum = Path(__file__).parent / "srf_eumetsat.csv"
    csv_data = pd.read_csv(file_srf_eum, header=0, index_col=0)
    
    if not (csv_data["id_EUMETSAT"] == id_sensor).any():
        warn(f"Sensor {id_sensor} is not present in srf_eumetsat.csv, "
             "and may be unavailable in EUMETSAT database.")

    # Sensor selection
    ds = xr.Dataset()
    ds.attrs["desc"] = f'Spectral response functions for {id_sensor}'
    ds.attrs["sensor"] = id_sensor

    urlpath = empty_link.format(id_sensor, id_sensor)

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
            sep='\s+',
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

    if 'olci' in ds.sensor.lower():
        # Special case OLCI: provide a mapping of (band, camera, ccd_col) to the
        # corresponding variable name
        # https://nwp-saf.eumetsat.int/downloads/rtcoef_rttov13/ir_srf/rtcoef_sentinel3_1_olci_srf.html
        assert (315 in ds) and (316 not in ds)
        ds["id"] = xr.DataArray(
            np.arange(1, 316).reshape((21, 5, 3)),
            dims=("band_id", "camera", "ccd_col"),
            coords={
                "band_id": np.arange(1, 22),
                "camera": ["FM5R", "FM9", "FM7", "FM10", "FM8"],
                "ccd_col": [10, 374, 730],
            },
        )

    return ds


@filegen(if_exists='skip')
def download_extract(directory: Path, url: str, verbose: bool = False):
    """
    Download a tar.gz file, and extract it to `directory`
    """
    with TemporaryDirectory() as tmpdir:
        tar_gz_path = download_url(url, tmpdir, verbose=verbose)
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
        if band == "id":
            # "id" is a special variable mapping various parameters to a variable name
            # thus it should not be considered in integrate_srf
            continue

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


def select(srf: xr.Dataset, **kwargs) -> xr.Dataset:
    """
    Apply an index selection on the srf object.

    If srf contains an "id" variable, it is used to trim down the variables.

    The kwargs are used for the selection based on the dimensions present in srf.

    Example: select(srf, camera=1)
        returns a subset of the srf for camera 1.
    """
    if "id" in srf:
        ids = srf.id.sel(**kwargs)
        list_vars = list(ids.values)
        return srf[list_vars]
    else:
        return srf.sel(**kwargs)


class _Func_ODR:
    def simpson(y, x=None):
        return simpson(y, x=x)

    def trapz(y, x=None):
        return np.trapz(y, x=x)


def to_tuple(
    srf: xr.Dataset, minT: float = 0.005, func_odr: Callable = _Func_ODR.simpson
) -> tuple:
    """
    Convert an srf to a tuple for compatibility with codes that use such srf
    representation (from smaclib)

    Arguments:
        srf: a dataset coming from get_SRF
        minT: threshold for minimum transmission subsetting.
            0.005 by default (legacy)
        func_odr: the function used for ODR integration, can either be _Func_ODR.simpson
            or _Func_ODR.trapz, _Func_ODR.simpson by default (legacy)

    returns:
        (wvn_limits, wvl_limits, fwhm, wvl_central, rod_effective, srf_wvl, rsrf)
        with wvn in cm-1, wvl in nm, fwhm in nm, wvl_central in nm,
        SRF weighted Rayleigh optical depth, reference wavelegnth of the rsrf in nm, rsrf, name (string)
    """
    name = list(srf.coords)
    srf_data = []
    srf_wvl = []
    fwhm = []
    central_wvl = []
    xLimits = []
    ODR = []
    
    keys_val = list(srf.keys())
    
    for var_name in keys_val:
        if len(srf[var_name].dims) != 1:
            print(f"Variable '{var_name}' has more than one coord: {srf[var_name].dims}")
    
    for i in range(len(keys_val)):
        coord = srf[keys_val[i]].coords.dims[0]
        is_crois = None
        if srf[coord][0] < srf[coord][1] :
            is_crois = True
        else :
            is_crois = False
        
        
        # SRF compute
        srf_ = np.array(srf[keys_val[i]] if is_crois else srf[keys_val[i]][::-1])  # croiss
        srf_ = srf_ / np.nanmax(srf_)
        ok = srf_ > minT  # subset only minimum transmission
        srf_ = srf_[ok]
        # SRF_WVL compute
        srf_wvl_ = np.array(srf[coord] if is_crois else srf[coord][::-1])  # croiss
        print("avant ok\n", srf_wvl_)
        srf_wvl_ = srf_wvl_[ok]
        print("après ok\n", srf_wvl_)

        # XLIMITS
        xLimits.append([1e7 / np.nanmax(srf_wvl_) + 1.0, 1e7 / np.nanmin(srf_wvl_) - 1.0]) #1e7 pour passer de nm à cm-1

        # fwhm and central_wvl computes
        mask05 = srf_ > 0.5
        high_end = srf_wvl_[mask05][-1]
        low_end = srf_wvl_[mask05][0]

        # FWHM
        fwhm.append(high_end - low_end)

        # CENTRAL_WVL
        central_wvl.append((high_end + low_end) * 0.5)

        # SRF & SRF_WVL attr
        srf_data.append(srf_)
        srf_wvl.append(srf_wvl_)

    # ODR
    for w, s in zip(srf_wvl, srf_data):
        od = np.squeeze(rod(w * 1e-3, np.array(400), 45.0, 0.0, 1013.25))
        integrate = func_odr(s * od, x=w)
        normalize = func_odr(s, x=w)
        ODR.append(integrate / normalize)

    return (
        np.array(xLimits), #cm-1
        1e7 / np.array(xLimits)[:, ::-1],
        fwhm,
        central_wvl,
        np.array(ODR),
        srf_wvl,
        srf_data,
        np.array(name),
    )



def get_SRF_landsat8_oli() -> xr.Dataset:
    """
    Get SRF for LANDSAT8 OLI
    """
    # Default directory
    dir_SRFs = env.getdir("DIR_STATIC") / "srf"
    
    fsrf = download_url(
        "https://docs.hygeos.com/s/i4wxHyoErgc8ZXX/download/Ball_BA_RSR.v1.2.xlsx",
        dir_SRFs,
    )
    data = pd.read_excel(fsrf, sheet_name="Band summary")
    bandnames = data["Band"][1:]
    ds = xr.Dataset()
    index = 1
    for band in bandnames:
        if band == "CA":
            band = "CoastalAerosol"
        data = pd.read_excel(fsrf, sheet_name=band)
        srf_ = np.array(data["BA RSR [watts]"])[::-1].astype(np.float64)
        srf_wvl_ = np.array(data["Wavelength"])[::-1].astype(np.float64)
        name_wvl = f"wav_{index}"
        dsb = xr.Dataset(
            {
                index: (
                    (name_wvl),
                    srf_,
                ),
            },
            coords={name_wvl: srf_wvl_},
        )
        dsb[name_wvl].attrs["units"] = "nm"
        ds = ds.merge(dsb)
        index += 1
    return ds


def get_SRF_misr() -> xr.Dataset:
    """
    Get SRF for MISR
    """
    # Default directory
    dir_SRFs = env.getdir("DIR_STATIC") / "srf"
    
    fsrfs = download_url(
        "https://docs.hygeos.com/s/z3TQoB7dEx9PzMq/download/MISR_SRF.txt", dir_SRFs
    )
    names = open(fsrfs, "r").readlines(1000)[17][:-1].split(",")
    data = np.loadtxt(fsrfs, skiprows=18, delimiter=",")
    nb = data[0, 2:].size

    ds = xr.Dataset()

    srf_wvl = np.array(data[:, 0])[::-1].astype(np.float64)
    
    name_wvl = names[0].strip()
    for i in np.arange(nb):
        srf_ = np.array(data[:, i + 2])[::-1].astype(np.float64)
        dsb = xr.Dataset(
            {
                names[i+1].strip(): (
                    (name_wvl),
                    srf_,
                ),
            },
            coords={name_wvl: srf_wvl},
        )
        ds = ds.merge(dsb)
    ds[name_wvl].attrs["units"] = "nm"
    return ds


def get_SRF_probav() -> xr.Dataset:
    """
    Get SRF for Proba-V
    """
    # Default directory
    dir_SRFs = env.getdir("DIR_STATIC") / "srf"
    
    fsrfs = download_url(
        "https://docs.hygeos.com/s/YyH4gc5HR5F9iKo/download/VGT_SRF.XLSX", dir_SRFs
    )
    data = pd.read_excel(fsrfs, sheet_name="Proba-V")
    data.rename(index=str, columns={"NIR  CENTER": "NIR CENTER"}, inplace=True)

    ds = xr.Dataset()

    for band in ["BLUE", "RED", "NIR", "SWIR"]:
        name_wvl = "wvl_{}".format(band)    
        srf_wvl_ = np.array(data[name_wvl].values).astype(np.float64)
        list_camera = ["CENTER", "LEFT", "RIGHT"]
        srf_ = []
        for dir in list_camera:
            name_srf = "{} {}".format(band, dir)
            srf_.append(np.array(data[name_srf].values).astype(np.float64))
        dsb = xr.Dataset(
            {
                band: (
                    ("camera", name_wvl),
                    srf_,
                ),
            },
            coords={"camera": list_camera ,name_wvl: srf_wvl_},
        )
        dsb[name_wvl].attrs["units"] = "nm"
        ds = ds.merge(dsb)
    return ds


def get_SRF_vgt(sensor: str) -> xr.Dataset:
    """
    Get SRF for VGT

    sensor: VGT1 or VGT2
    """
    # Default directory
    dir_SRFs = env.getdir("DIR_STATIC") / "srf"
    
    fsrfs = download_url(
        "https://docs.hygeos.com/s/YyH4gc5HR5F9iKo/download/VGT_SRF.XLSX", dir_SRFs
    )
    data = pd.read_excel(fsrfs, sheet_name=sensor)

    ds = xr.Dataset()
    index = 1

    for band in ["BLUE", "RED", "NIR", "SWIR"]:
        name_srf = "{} {}".format(band, sensor)
        if sensor == "VGT1":
            srf_wvl_ = np.array(data["wavelength"].values * 1e3).astype(np.float64)
            srf_ = np.array(data[name_srf].values).astype(
                np.float64
            )
        elif sensor == "VGT2":
            srf_wvl_ = np.array(data["wavelength"].values).astype(np.float64)
            srf_ = np.array(data[name_srf].values).astype(
                np.float64
            )
        else:
            raise ValueError

        dsb = xr.Dataset(
            {
                name_srf: (
                    ("wavelength"),
                    srf_,
                ),
            },
            coords={"wavelength": srf_wvl_},
        )
        ds = ds.merge(dsb)
        index += 1
    ds["wavelength"].attrs["units"] = "nm"
    return ds


def get_SRF_olci(platform: str) -> xr.Dataset:
    """
    Get SRF for OLCI

    platform: one of:
        "SENTINEL3_1", "SENTINEL3-A", 
        "SENTINEL3_2"  "SENTINEL3-B"
    """
    # Default directory
    dir_SRFs = env.getdir("DIR_STATIC") / "srf"

    if platform == "SENTINEL3_1":
        fsrf = h5py.File(
            download_url(
                "https://docs.hygeos.com/s/BTRbcPoiqnwrrTj/download/S3A_OL_SRF_20160713_mean_rsr.nc4",
                dir_SRFs,
            ),
            "r",
        )
    else:
        fsrf = h5py.File(
            download_url(
                "https://docs.hygeos.com/s/YNfiCaBZXfsYfQB/download/S3B_OL_SRF_0_20180109_mean_rsr.nc4",
                dir_SRFs,
            ),
            "r",
        )
    central_wvl_i = np.copy(fsrf["srf_centre_wavelength"])
    srf_wvl_i = np.copy(fsrf["mean_spectral_response_function_wavelength"])
    srf_i = np.copy(fsrf["mean_spectral_response_function"])
    fsrf.close()

    ds = xr.Dataset()
    index = 1

    for i in np.arange(len(central_wvl_i)):
        srf_ = np.array(srf_i[i, :])[::-1].astype(np.float64)  # normalize SRF
        srf_wvl_ = np.array(srf_wvl_i[i, :])[::-1].astype(np.float64)
        dsb = xr.Dataset(
            {
                index: (
                    (f"wav_{index}"),
                    srf_,
                ),
            },
            coords={f"wav_{index}": srf_wvl_},
        )
        dsb[f"wav_{index}"].attrs["units"] = "nm"
        ds = ds.merge(dsb)
        index += 1
    return ds


def get_SRF_meris() -> xr.Dataset:
    """
    Get SRF for MERIS
    """
    # Default directory
    dir_SRFs = env.getdir("DIR_STATIC") / "srf"
    
    fsrfs = pd.read_excel(
        download_url(
            "https://docs.hygeos.com/s/XTJCtxjDr7YmMjY/download/MERIS_NominalSRF_Model2004.xls",
            dir_SRFs,
        ),
        sheet_name="NominalSRF Model2004",
        skiprows=1,
    )
    ds = xr.Dataset()
    index = 1

    for i in range(15):
        if i == 0:
            st = ""
        else:
            st = "." + str(i)
        srf_wvl_ = np.array(fsrfs["wavelength" + st])[::-1].astype(np.float64)
        srf_ = np.array(fsrfs["SRF" + st])[::-1].astype(np.float64)

        dsb = xr.Dataset(
            {
                index: (
                    (f"wav_{index}"),
                    srf_,
                ),
            },
            coords={f"wav_{index}": srf_wvl_},
        )
        dsb[f"wav_{index}"].attrs["units"] = "nm"
        ds = ds.merge(dsb)
        index += 1
    return ds


def get_SRF_msi(platform : str) -> xr.Dataset:
    """
    Get SRF for Sentinel-2 MSI

    platform: "S2A", "S2B", "S2C"
    """
    # Default directory
    dir_SRFs = env.getdir("DIR_STATIC") / "srf"
    
    fsrfs = download_url(
        "https://docs.hygeos.com/s/Q66yAcaseLaJXzj/download/COPE-GSEG-EOPG-TN-15-0007%20-%20Sentinel-2%20Spectral%20Response%20Functions%202024%20-%204.0.xlsx",
        dir_SRFs,
    )
    data = pd.read_excel(fsrfs, sheet_name="Spectral Responses ({})".format(platform))
    bands = ["B{:d}".format(i + 1) for i in range(8)] + [
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
    ]

    ds = xr.Dataset()
    index = 1

    np.set_printoptions(threshold=np.inf)
    srf_wvl_ = np.array(data["SR_WL"].values).astype(np.float64)
    for band in bands:
        srf_ = np.array(data["{}_SR_AV_{}".format(platform, band)].values).astype(
            np.float64
        )
        dsb = xr.Dataset(
            {
                index: (
                    ("wav"),
                    srf_,
                ),
            },
            coords={"wav": srf_wvl_},
        )
        
        ds = ds.merge(dsb)
        index += 1
    ds["wav"].attrs["units"] = "nm"
    return ds