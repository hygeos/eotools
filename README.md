# eotools

Various tools and modules for satellite data management and atmospheric correction.

## Installation

To install or update this module from github in your current python environment, use the following command:

```
pip install git+https://github.com/hygeos/eotools.git
```

Required dependencies include [hygeos-core](https://pypi.org/project/hygeos-core/) (imported as `hygeos_core`) and [luts](https://pypi.org/project/luts/). To also install [eoread](https://github.com/hygeos/eoread) (level-1 readers and ancillary data), use the `git` extra:

```
pip install "eotools[git] @ git+https://github.com/hygeos/eotools.git"
```

To install it from a cloned folder, in editable mode:

```
pip install --no-deps -e <dir/to/eotools/>
```


## Documentation

Full documentation is available at: https://hygeos.github.io/eotools/
