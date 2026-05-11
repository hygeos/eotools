from pathlib import Path
from typing import Dict
import pandas as pd
import xarray as xr
import numpy as np

from pyhdf.SD import SD
from core.tools import datetime
from core.files.fileutils import mdir
from core.network.download import download_url
from eotools.gaseous_absorption import get_absorption, get_transmission_coeffs
from eotools.srf import integrate_srf
from eotools.units import convert
from core import env
from core.tools import MapBlocksOutput
from core.process.blockwise import BlockProcessor, CompoundProcessor, Var


class Gaseous_correction(CompoundProcessor):
    """Gaseous correction module

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to which the gaseous correction is applied.
        The bands of `ds` shall be contained in `srf`.
    srf : xr.Dataset
        The sensor spectral response function (SRF).
    input_var: str
        Name of the input variable in `ds` (Top of atmosphere reflectance)
    ouput_var: str
        Name of the output variable in `ds`
    spectral_dim: str
        Name of the spectral dimension in input_var and output_var
    dir_common : Optional[Path]
        Path to the `common` directory.

    Example
    -------
    Apply gaseous correction to `l1`

    >>> Gaseous_correction(l1).apply()
    """                 
    def __init__(self,
                 ds: xr.Dataset,
                 srf: xr.Dataset | None = None,
                 input_var: str='rho_toa',
                 ouput_var: str='rho_gc',
                 spectral_dim: str='bands',
                 K_NO2: Dict | None = None,
                 K_OZ: Dict | None = None,
                 no2_correction: str = "legacy",
                 gas_correction: str = "o3_legacy",
                 bands_sel_ckdmip: list | slice | None = None,
                 **kwargs
                 ):

        self.no2_correction = no2_correction
        self.gas_correction = gas_correction
        self.ds = ds
        self.spectral_dim = spectral_dim
        self.bands = list(ds[spectral_dim].data)
        self.input_var = input_var
        self.output_var = ouput_var
        self.model = MapBlocksOutput([
            Var(ouput_var, dtype='float32', dims=('y', 'x', 'bands'))
        ])

        try:
            # image mode: single date for the whole image
            self.datetime = datetime(ds)
        except AttributeError:
            # extraction mode: per-pixel date
            self.datetime = None

        dir_common = mdir(env.getdir("DIR_STATIC") / "common")
        
        # Initialize sub classes
        processors: list[BlockProcessor] = [
            Init_rho_gc(input_var),
            Init_air_mass(ds),
        ]
        if no2_correction == "legacy":
            self.corr_NO2 = Gas_correction_NO2(ds, srf, dir_common, K_NO2, spectral_dim)
            processors.append(self.corr_NO2)
        if gas_correction == 'o3_legacy':
            self.corr_O3 = Gas_correction_O3(ds, srf, dir_common, K_OZ, spectral_dim)
            processors.append(self.corr_O3)
        elif gas_correction == 'ckdmip':
            assert bands_sel_ckdmip is not None
            self.corr_ckdmip = Gas_correction_CKDMIP(ds, bands_sel=bands_sel_ckdmip)
            processors.append(self.corr_ckdmip)
        else:
            raise ValueError

        super().__init__(processors)

    def run(self, bands, Rtoa, mus, muv,
            ozone_dobson, tcwv_kgm2, ssp_hPa, latitude, longitude,
            flags, 
            datetime):
        """
        Apply gaseous correction to Rtoa (ozone, NO2)

        ozone_dobson : total column in Dobson Unit (DU)
        tcwv_kgm2: total column water vapour (kg/m2)
        ssp_hPa : sea surface pressure (hPa)
        """
        Rtoa_gc = Rtoa.copy()
        ok = ~np.isnan(latitude)

        # TODO: nightpixels & invalid pixels
        air_mass = 1/muv + 1/mus

        # O3 correction
        if self.gas_correction == 'o3_legacy':
            self.corr_O3.run(bands, Rtoa_gc, ok, air_mass, ozone_dobson)
        elif self.gas_correction == 'ckdmip':
            self.corr_ckdmip.run(bands, Rtoa_gc, ok, air_mass, ozone_dobson, tcwv_kgm2, ssp_hPa)

        # NO2 correction
        if self.no2_correction == "legacy":
            self.corr_NO2.run(
                bands,
                Rtoa_gc,
                ok,
                air_mass,
                latitude,
                longitude,
                datetime,
            )

        return Rtoa_gc

    def apply(self, method='map_blocks'):
        """Apply gaseous to current dataset `ds`

        This creates the variable `rho_gc` in the current dataset
        """        
        ds = self.ds
        date = datetime(ds)

        # TODO: use pint here
        # TODO: move this to "map_blocks section"
        if ds.total_column_ozone.units in ['Kg.m-2', 'kg m**-2', 'kg.m-2']:
            total_ozone = ds.total_column_ozone / 2.1415e-5  # convert kg/m2 to DU
        else:
            total_ozone = ds.total_column_ozone
            assert ds.total_column_ozone.units in ['DU','Dobsons', 'Dobson']
        

        if self.gas_correction == 'ckdmip':
            tcwv = convert(ds['total_column_water_vapour'], 'g/cm²')
            ssp_hPa = convert(ds['sea_level_pressure'], 'hPa')
        else:
            tcwv = None
            ssp_hPa = None

        rho_toa = ds[self.input_var].chunk({self.spectral_dim: -1})

        if method == 'map_blocks':
            ds_out = xr.map_blocks(
                self.apply_block,
                xr.Dataset(
                    {
                        'rho_toa': rho_toa,
                        'mus': ds.mus,
                        'muv': ds.muv,
                        'total_ozone': total_ozone,
                        'tcwv': tcwv,
                        'ssp': ssp_hPa,
                        'latitude': ds.latitude,
                        'longitude': ds.longitude,
                        'flags': ds.flags,
                    }
                ),
                template=self.model.template(ds),
                kwargs={'dt': date},
            )
            ds[self.output_var] = ds_out[self.output_var]

        elif method == 'blockwise':
            ds_out = self.map_blocks(ds)
            for x in self.output_vars():
                ds[x] = ds_out[x]
        else:
            raise ValueError(method)

    def apply_block(self, ds: xr.Dataset, dt):
        Rtoa_gc = self.run(
            ds[self.spectral_dim].data,
            ds.rho_toa.transpose(..., 'bands').data,
            ds.mus.data,
            ds.muv.data,
            ds.total_ozone.data,
            ds.tcwv.data,
            ds.ssp.data,
            ds.latitude.data,
            ds.longitude.data,
            ds.flags.data,
            datetime=dt
        ).astype('float32')
        out = xr.Dataset()
        out[self.output_var] = xr.DataArray(
            Rtoa_gc,
            dims=ds.latitude.dims + ("bands",),
            coords={"bands": ds.bands},
        )
        return self.model.conform(out)

class Init_rho_gc(BlockProcessor):
    def __init__(self, input_var: str):
        self.input_var = input_var

    def input_vars(self) -> list[Var]:
        return [Var(self.input_var)]

    def created_vars(self) -> list[Var]:
        return [Var("rho_gc")]
    
    def auto_template(self) -> bool:
        return True

    def process_block(self, block: xr.Dataset):
        block["rho_gc"] = block[self.input_var].copy(deep=True)


class Init_air_mass(BlockProcessor):
    def __init__(self, ds: xr.Dataset):
        self.activate = "air_mass" not in ds

    def input_vars(self) -> list[Var]:
        return [Var("mus"), Var("muv")] if self.activate else []

    def created_vars(self) -> list[Var]:
        return [Var("air_mass")]
    
    def auto_template(self) -> bool:
        return True

    def process_block(self, block: xr.Dataset):
        block['air_mass'] = 1/block.muv + 1/block.mus



class Gas_correction_O3(BlockProcessor):
    """
    Ozone absorption correction module
    """

    def __init__(
        self,
        ds: xr.Dataset,
        srf: xr.Dataset | None,
        dir_common: Path,
        K_OZ: Dict | None = None,
        spectral_dim: str='bands',
    ):
        self.spectral_dim = spectral_dim
        self.bands = list(ds[spectral_dim].data)

        if K_OZ is not None:
            # K_OZ is provided explicitly
            self.K_OZ = xr.Dataset()
            for k, v in K_OZ.items():
                self.K_OZ[k] = v
        else:
            # K_OZ calculated from SRF

            # load absorption rate for each gas
            k_oz_data  = get_absorption('o3', dirname=dir_common)

            if srf is not None: # K_OZ and K_NO2 are calculated from srf
                for b in self.bands:
                    assert b in srf

                self.K_OZ = integrate_srf(srf, k_oz_data, resample='x')

                # consistency checking: check that both wavc and ds.bands are sorted
                # wavc = integrate_srf(srf, lambda x: x)
                # assert (np.diff(ds.bands) > 0).all()
                # assert (np.diff(list(wavc.values())) > 0).all()
        
            else: # K_OZ is calculated from central wavelengths
                self.K_OZ = xr.Dataset()
                for i, k in enumerate(k_oz_data.interp(wav=ds.cwav).values):
                    self.K_OZ[self.bands[i]] = k
    
    def run(self, bands, Rtoa_gc, ok, air_mass,
            ozone):
        """
        Apply O3 transmission

        ozone : total column in Dobson Unit (DU)
        """

        # ozone correction
        for i, b in enumerate(bands):

            if not Rtoa_gc.size:
                break 

            tauO3 = self.K_OZ[b].values * ozone[ok] * 1e-3  # convert from DU to cm*atm

            # ozone transmittance
            trans_O3 = np.exp(-tauO3 * air_mass[ok])

            # TODO: check this
            Rtoa_gc[ok, i] /= trans_O3

        return Rtoa_gc.astype('float32')
    
    def input_vars(self) -> list[Var]:
        return [
            Var("air_mass"),
            Var("total_column_ozone"),
            Var("latitude"),
            Var("longitude"),
        ]

    def modified_vars(self) -> list[Var]:
        return [Var('rho_gc')]
    
    def process_block(self, block: xr.Dataset):
        ok = ~np.isnan(block.latitude.data)
        ozone_dobson = convert(block.total_column_ozone, 'Dobson').data
        self.run(
            block[self.spectral_dim].data,
            block.rho_gc.transpose(..., 'bands').data,
            ok,
            block.air_mass.data,
            ozone_dobson
            )


class Gas_correction_NO2(BlockProcessor):
    def __init__(
        self,
        ds: xr.Dataset,
        srf: xr.Dataset | None,
        dir_common: Path,
        K_NO2: Dict | None = None,
        spectral_dim: str='bands',
    ):
        self.spectral_dim = spectral_dim
        self.bands = list(ds[spectral_dim].data)
        self.datetime = datetime(ds)

        # Collect auxilary data
        self.no2_climatology = download_url(
            'https://github.com/hygeos/eotools/releases/download/root/no2_climatology.hdf',
            dir_common, verbose=False,
        )
        self.no2_frac200m = download_url(
            'https://github.com/hygeos/eotools/releases/download/root/trop_f_no2_200m.hdf',
            dir_common, verbose=False,
        )

        if K_NO2 is not None:
            # K_NO2 is provided explicitly
            self.K_NO2 = xr.Dataset()
            for k, v in K_NO2.items():
                self.K_NO2[k] = v
        else:
            # load absorption rate for each gas
            k_no2_data = get_absorption('no2', dirname=dir_common)

            if srf is not None:
                for b in self.bands:
                    assert b in srf

                self.K_NO2 = integrate_srf(srf, k_no2_data, resample='x')

                # consistency checking: check that both wavc and ds.bands are sorted
                # wavc = integrate_srf(srf, lambda x: x)
                # assert (np.diff(ds.bands) > 0).all()
                # assert (np.diff(list(wavc.values())) > 0).all()
        
            else: # K_OZ and K_NO2 are calculated from central wavelengths
                self.K_NO2 = xr.Dataset()
                for i, k in enumerate(k_no2_data.interp(wav=ds.cwav).values):
                    self.K_NO2[self.bands[i]] = k

    def run(
        self,
        bands,
        Rtoa_gc,
        ok,
        air_mass,
        latitude,
        longitude,
        datetime,
    ):
        """
        Apply NO2 transmission
        """

        # NO2 correction
        no2_frac, no2_tropo, no2_strat = self.get_no2(latitude, longitude, datetime)
        no2_tr200 = no2_frac * no2_tropo
        no2_tr200[no2_tr200 < 0] = 0

        for i, b in enumerate(bands):

            if not Rtoa_gc.size:
                break 

            k_no2 = self.K_NO2[b].values

            a_285 = k_no2 * (1.0 - 0.003*(285.0-294.0))
            a_225 = k_no2 * (1.0 - 0.003*(225.0-294.0))

            tau_to200 = a_285*no2_tr200 + a_225*no2_strat

            t_no2 = np.exp(-tau_to200[ok] * air_mass[ok])

            Rtoa_gc[ok, i] /= t_no2
        

    def read_no2_data(self, month):
        '''
        read no2 data from month (1..12) or for all months if month < 0

        shape of arrays is (month, lat, lon)
        '''
        hdf1 = SD(str(self.no2_climatology))
        hdf2 = SD(str(self.no2_frac200m))

        if month < 0:
            months = range(1, 13)
            nmo = 12
        else:
            months = [month]
            nmo = 1

        self.no2_total_data = np.zeros((nmo,720,1440), dtype='float32')
        self.no2_tropo_data = np.zeros((nmo,720,1440), dtype='float32')
        self.no2_frac200m_data = np.zeros((90,180), dtype='float32')

        for i, m in enumerate(months):
            # read total and tropospheric no2 data
            self.no2_total_data[i,:,:] = hdf1.select('tot_no2_{:02d}'.format(m)).get()

            self.no2_tropo_data[i,:,:] = hdf1.select('trop_no2_{:02d}'.format(m)).get()

        # read fraction of tropospheric NO2 above 200mn
        self.no2_frac200m_data[:,:] = hdf2.select('f_no2_200m').get()

        hdf1.end()
        hdf2.end()

    def get_no2(self, latitude, longitude, datetime):
        """
        returns no2_frac, no2_tropo, no2_strat at the pixels coordinates
        (latitude, longitude)
        """
        ok = ~np.isnan(latitude) & ~np.isnan(longitude)

        # get month
        if not hasattr(datetime, "month"):
            mon = -1
            imon = pd.DatetimeIndex(datetime).month - 1
        else:
            mon = datetime.month
            imon = 0

        try:
            self.no2_tropo_data
        except Exception:
            self.read_no2_data(mon)

        # coordinates of current block in 1440x720 grid
        ilat = (4*(90 - latitude)).astype('int')
        ilon = (4*longitude).astype('int')
        ilon[ilon<0] += 4*360
        ilat[~ok] = 0
        ilon[~ok] = 0

        no2_tropo = self.no2_tropo_data[imon,ilat,ilon]*1e15
        no2_strat = (self.no2_total_data[imon,ilat,ilon]
                     - self.no2_tropo_data[imon,ilat,ilon])*1e15

        # coordinates of current block in 90x180 grid
        ilat = (0.5*(90 - latitude)).astype('int')
        ilon = (0.5*longitude).astype('int')
        ilon[ilon<0] += 180
        ilat[~ok] = 0
        ilon[~ok] = 0
        no2_frac = self.no2_frac200m_data[ilat,ilon]

        return no2_frac, no2_tropo, no2_strat

    def input_vars(self) -> list[Var]:
        return [
            Var('latitude'),
            Var('longitude'),
            Var('air_mass'),
        ]
        
    def modified_vars(self) -> list[Var]:
        return [Var('rho_gc')]
    
    def process_block(self, block: xr.Dataset):
        ok = ~np.isnan(block.latitude.data)
        self.run(
            block[self.spectral_dim].data,
            block.rho_gc.transpose(..., 'bands').data,
            ok,
            block.air_mass.data,
            block.latitude.data,
            block.longitude.data,
            self.datetime,
        )


class Gas_correction_CKDMIP(BlockProcessor):
    def __init__(self, ds: xr.Dataset, bands_sel: list | slice):
        self.coeffs = get_transmission_coeffs((ds.platform, ds.sensor)).sel(
            bands=bands_sel
        )

        self.nbands = len(self.coeffs.bands)
    
    def run(
        self, bands, Rtoa_gc, ok, air_mass, ozone, tcwv, sea_surface_pressure
    ):
        """
        Ozone in Dobson
        tcwv in g/cm2
        sea_surface_pressure in hPa
        """

        assert self.nbands == Rtoa_gc.shape[-1]
        
        for gas in self.coeffs.gas:
            c = self.coeffs.sel(gas=gas)
            if gas == 'O3':
                assert c.U_units.values == "Dobson"
                x = air_mass[ok] * ozone[ok] / c.U0.values
            elif gas == 'H2O':
                assert c.U_units.values == "g/cm²"
                x = air_mass[ok] * tcwv[ok] / c.U0.values
            else:
                x = air_mass[ok] * sea_surface_pressure[ok] / c.P0.values
            T = np.exp(-c.a.values * x[:,None]**c.n.values)
            
            Rtoa_gc[ok,:] /= T

    def input_vars(self) -> list[Var]:
        return [
            Var("latitude"),
            Var("longitude"),
            Var("air_mass"),
            Var("total_column_water_vapour"),
            Var("total_column_ozone"),
            Var("sea_level_pressure"),
        ]
    def modified_vars(self) -> list[Var]:
        return [Var('rho_gc')]
        
    def process_block(self, block: xr.Dataset):
        ok = ~np.isnan(block.latitude.data)
        ozone_dobson = convert(block['total_column_ozone'], 'Dobson')
        tcwv = convert(block['total_column_water_vapour'], 'g/cm²')
        ssp_hPa = convert(block['sea_level_pressure'], 'hPa')
        self.run(
            block['bands'].data,
            block.rho_gc.transpose(..., 'bands').data,
            ok,
            block.air_mass.data,
            ozone_dobson.data,
            tcwv.data,
            ssp_hPa.data,
        )



