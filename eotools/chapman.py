import numpy as np
from scipy.special import erfc


def chapman(mu, X=6730.0 / 8.0, method="d"):
    """
    Chapman function

    Methods "d" (Titheridge (1988, 2000)) and "e" (Kocifaj (1996); Schüler (2012))
    See https://earth-planets-space.springeropen.com/articles/10.1186/s40623-021-01435-y/tables/1

    Arguments:
        mu : cos of the Zenith angle at the point of interest
        X : ratio of altitude at the point of interest (from Earths centre, default
            RTER=6730km) to scale height of the atmospheric absorber (default H0=8km,
            Rayleigh scale height)
    """

    if method == "e":
        ct2 = mu**2
        st2 = 1.0 - ct2
        return 0.5 * (
            mu
            + (
                np.sqrt(np.pi * X / 2.0)
                * erfc(np.sqrt(X / 2.0) * mu)
                * np.exp(X / 2.0 * ct2)
                * (1.0 / X + 1 + st2)
            )
        )

    elif method == "d":
        dzeta = np.arccos(mu)
        A = 1.0123 - 1.454 / np.sqrt(X)
        d = 3.88 * X ** (-1.143) * (1.0 / np.cos(A * dzeta) - 0.834)
        return 1.0 / np.cos(dzeta - d)