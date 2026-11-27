# Changelog

## v1.0.0 (2026-11-27)

### Breaking changes
- `core` replaced by [hygeos-core](https://pypi.org/project/hygeos-core/) (`>=2.0.2,<3`, imported as `hygeos_core`).
- `eoread` is now an optional dependency (`eoread>=2,<3`, extra `eoread`), installed from PyPI; the `git` extra is removed. Support for eoread < 2 is dropped.
- Required dependencies: `hygeos-core`, `luts` (`>=1,<2`), and the previously undeclared `numpy`, `pandas`, `scipy`, `dask`, `pyhdf`, `openpyxl`. `sand` is no longer a dependency.
- Rayleigh, gaseous correction, glint, DEM, water mask, cloud masks and pansharpening reimplemented as `BlockProcessor`s.

### Added
- Rayleigh correction processor (`process_rayleigh`, `process_rayleigh_dataset`, CLI `python -m eotools.rayleigh`), with per-sensor configuration, diffuse transmittance and sun glint corrections, custom Rayleigh optical depth, interpolation of valid pixels only.
- CKDMIP-based gaseous absorption correction, fitted from gatiab results.
- Per-band viewing geometry (view angles with a `bands` dimension).
- DEMs: `CopernicusDEM` (per-pixel tile lookup), `GTOPO30`, `ZeroAltitude`, DEM factory class.
- Water masks: Global Surface Water (`GSW`, `GSWLandMask`).
- SRFs: S2C, NASA sensors, `get_SRF_olci_single`, `get_SRF_gaussian`, `get_SRF_sgli`, multidimensional SRF integration.
- Modules `geometry`, `toa_reflectance`, `units`; Chapman function (method f by default).
- Quarto documentation, deployed with GitHub Actions.

### Fixed
- Sun-earth distance correction in `Init_rho_toa`, `t_d` interpolation, `calc_odr` surface pressure, NaN handling in SRFs, global average Earth radius in `chapman`.

## v0.1.0 (2025-11-13)

Initial version.
