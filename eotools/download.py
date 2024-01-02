from pathlib import Path
import subprocess
from .fileutils import filegen

def download_url(url, dirname, wget_opts='',
                 check_function=None,
                 verbose=True,
                 **kwargs
                 ):
    """
    Download `url` to `dirname` with wget

    Options `wget_opts` are added to wget
    Uses a `filegen` wrapper
    Other kwargs are passed to `filegen` (lock_timeout, tmpdir, if_exists)

    Returns the path to the downloaded file
    """
    target = Path(dirname)/(Path(url).name)
    if verbose:
        print('Downloading:', url)
        print('To: ', target)
    
    @filegen(**kwargs)
    def download_target(path):
        cmd = f'wget {wget_opts} {url} -O {path}'
        # Detect access problem
        if subprocess.call(cmd.split()):
            raise RuntimeError(f'Authentification issue : "{cmd}"')

        if check_function is not None:
            check_function(path)

    download_target(target)

    return target