import importlib
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Union
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
    ds: xr.Dataset,
    srf_getter: str | None = None,
    srf_getter_arg: str | None = None,
    fuzzy: bool = True,
    **kwargs
) -> xr.Dataset:
    """
    Get a SRF for the product `ds`

    Args:
        ds: a dataset, which contains platform and sensor attributes (unless
            srf_getter/srf_getter_arg is specified)
        srf_getter (str): name of custom function to override the function for SRF reading.
            Example: "eotools.srf.get_SRF_eumetsat"
            If None, `srf_getter` and `srf_getter_arg` are searched in the `srf.csv`
            file for the sensor/platform specified in `ds`.
        srf_getter_arg (str): argument passed to the srf_getter function
            Example: "dscovr_1_epic"
        fuzzy (bool): whether to apply fuzzy search to the sensor and platform (don't match
            case, ignore whitespaces, "-" and "_")

    Return a xr.Dataset with the SRF for each band, defined by default identifiers.
    """
    def apply_fuzzy(x):
        if fuzzy:
            return x.replace("-", "").replace("_", "").strip().lower()
        else:
            return x

    if srf_getter is None:
        platform = ds.platform
        sensor = ds.sensor
        
        # load CSV file
        file_srf = Path(__file__).parent / "srf.csv"
        csv_data = pd.read_csv(
            file_srf, header=0, index_col=False, dtype=str, comment="#"
        )

        csv_platform = csv_data["platform"].astype("str").apply(apply_fuzzy)
        csv_sensor = csv_data["sensor"].astype("str").apply(apply_fuzzy)

        eq = (csv_sensor == apply_fuzzy(sensor)) & (
            csv_platform == apply_fuzzy(platform)
        )

        if sum(eq) != 1:
            raise ValueError(
                f"Sensor {platform}/{sensor} is not present in srf.csv. Please provide"
                "your own srf_getter."
            )
        
        srf_getter = str(csv_data[eq].srf_getter.values[0])
        srf_getter_arg = str(csv_data[eq].srf_getter_arg.values[0])

    # Run the SRF getter
    p, m = srf_getter.rsplit(".", 1)
    mod = importlib.import_module(p)
    getter = getattr(mod, m)
    if srf_getter_arg == 'nan':
        srf = getter()
    else:
        srf = getter(srf_getter_arg)

    if "desc" not in srf.attrs:
        srf.attrs["desc"] = f'Spectral response functions for {platform}/{sensor}'
    if "sensor" not in srf.attrs:
        srf.attrs["sensor"] = sensor
    if "platform" not in srf.attrs:
        srf.attrs["platform"] = platform
    
    return srf


def get_SRF_eumetsat(id_sensor: str) -> xr.Dataset:
    """
    Download and read Specrtral Response Function (SRF) from EUMETSAT database
    -> https://nwp-saf.eumetsat.int/site/software/rttov/download/coefficients/spectral-response-functions/

    Args:
        id_sensor: identifier of the sensor/platform (as in the srf.csv file)

    Return a xr.Dataset with the SRF for each band, identified by integers from 1 to
    nbands
    """
    
    empty_link = "https://nwp-saf.eumetsat.int/downloads/rtcoef_info/visir_srf/rtcoef_{}_srf/rtcoef_{}_srf.tar.gz"

    # Default directory
    directory = env.getdir("DIR_STATIC") / "srf"

    # Load srf.csv to check the list of available sensors
    # and raise a warning if needed
    file_srf = Path(__file__).parent / "srf.csv"
    csv_data = pd.read_csv(
        file_srf, header=0, index_col=False, dtype=str, comment="#"
    )
    
    if not (csv_data["srf_getter_arg"] == id_sensor).any():
        warn(f"Sensor {id_sensor} is not present in srf.csv, "
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

    for i, filepath in enumerate(list_files):
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
        bid = i+1
        ds[bid] = xr.DataArray(
            srf["Response"].values,
            coords={f"wav_{bid}": srf["Wavelength"].values},
            attrs={"band_info": binfo.strip()},
        )
        ds[f"wav_{bid}"].attrs["units"] = "nm"

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


def nbands(srf: xr.Dataset) -> int:
    """
    Returns the number of bands of a SRF object
    """
    if "id" in srf:
        assert "band_id" in srf.id
        return len(srf.id.band_id)
    else:
        return len(srf)


def rename(
    srf: xr.Dataset,
    band_ids: List[int] | np.ndarray,
    check_nbands: bool = True,
    thres_check: float | None = 10.0,
) -> xr.Dataset:
    """
    Rename bands in a SRF object

    Args:
        srf: the input srf object, with arbitrary band names
        band_ids: the new band identifiers
        check_nbands: whether the number of bands passed as band_ids should match
            the number of srfs.
        thres_check: check that the integrated srf is within a distance of `thres_check`
            of band_id (assumed integer)
            If None, this test is disactivated.
    """
    if check_nbands:
        # check that the number of read bands matches `band_ids`
        assert nbands(srf) == len(band_ids)
    
    # rename input variables to band_ids
    in_vars = [k for k in list(srf) if k != "id"]
    srf = srf.rename(dict(zip(in_vars, band_ids)))

    # check consistency
    if (thres_check is not None):
        # check that the band id matches the srf
        cwav = integrate_srf(srf, lambda x: x)
        for bid in band_ids:
            diff = abs(float(bid) - cwav[bid])
            assert diff < thres_check, ("There might be an error when providing the "
                f"SRFs. A central wavelength of {cwav[bid]} was found for band {bid}")
    
    return srf


@filegen(if_exists='skip')
def download_extract(directory: Path, url: str, verbose: bool = False):
    """
    Download a tar.gz file, and extract it to `directory`
    """
    with TemporaryDirectory() as tmpdir:
        tar_gz_path = download_url(url, Path(tmpdir), verbose=verbose)
        with tarfile.open(tar_gz_path) as f:
            f.extractall(directory)


def integrate_srf(
    srf: xr.Dataset, x: Union[Callable, xr.DataArray],
    integration_function: Callable = simpson
) -> Dict:
    """
    Integrate the quantity x over each spectral band in srf

    If x is a Callable, it is assumed that it takes inputs as a unit of nm
    If x is a DataArray, it should have a dimension "wav" and an associated unit

    integration_function: can be one of:
        - np.trapz
        - scipy.integrate.simpson

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

        integrate = integration_function(srf_band.values * xx, x=wav)
        normalize = integration_function(srf_band.values, x=wav)
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


def plot_srf(srf: xr.Dataset):
    """
    Plot a SRF Dataset
    """
    from matplotlib import pyplot as plt
    plt.figure()
    for iband in srf.data_vars:
        if iband == 'id':
            continue
        srf[iband].plot(label=iband)
        for coord in srf[iband].coords:
            if srf[iband].coords[coord].ndim == 0:
                # ignore scalar coords
                continue
            assert "units" in srf[coord].attrs
    plt.title(srf.desc)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel("wavelength")
    plt.ylabel("SRF")
    plt.grid(True)


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

        ds[band] = xr.DataArray(
            srf_,
            dims=("camera", name_wvl),
            coords={"camera": list_camera, name_wvl: srf_wvl_},
        )
        ds[name_wvl].attrs["units"] = "nm"

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

    platform: one of "SENTINEL3_1", "SENTINEL3_2"
    """
    # Default directory
    dir_SRFs = env.getdir("DIR_STATIC") / "srf"

    url = {
        "SENTINEL3_1": "https://docs.hygeos.com/s/BTRbcPoiqnwrrTj/download/S3A_OL_SRF_20160713_mean_rsr.nc4",
        "SENTINEL3_2": "https://docs.hygeos.com/s/YNfiCaBZXfsYfQB/download/S3B_OL_SRF_0_20180109_mean_rsr.nc4",
    }[platform]
    fsrf = h5py.File(download_url(url, dir_SRFs), "r")
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