# eotools

Various tools and modules for satellite data management and atmospheric correction.

## Installation

From PyPI:

```
pip install eotools
```

or from conda-forge:

```
conda install -c conda-forge eotools
```

Required dependencies include [hygeos-core](https://pypi.org/project/hygeos-core/) (imported as `hygeos_core`) and [luts](https://pypi.org/project/luts/). To also install [eoread](https://pypi.org/project/eoread/) >= 2 (level-1 readers and ancillary data), use the `eoread` extra (`pip install "eotools[eoread]"`), or install `eoread` with conda.

To install the development version from GitHub:

```
pip install git+https://github.com/hygeos/eotools.git
```

To install it from a cloned folder, in editable mode:

```
pip install --no-deps -e <dir/to/eotools/>
```


## Documentation

Full documentation is available at: https://hygeos.github.io/eotools/
