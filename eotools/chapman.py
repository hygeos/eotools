import numpy as np
from scipy.special import erfc, erfcx
from math import pi


def chapman(mu, X=6371. / 8.0, method="f"):
    """
    Compute the optical slant air mass accounting for the earth's curvature
    using the Chapman function.

    Parameters
    ----------
    mu : float or array_like
        Cosine of the zenith angle at the point of interest.
    X : float, optional
        Ratio of altitude at the point of interest (from Earth's centre) to the
        scale height of the atmospheric absorber. Default is 6371/8 ≈ 796.375,
        corresponding to Earth's mean radius (6371 km) and Rayleigh scale height (8 km).
    method : {'d', 'e', 'f'}, optional
        Method for computation. Default is 'f'.
        
        - 'd': Titheridge (1988, 2000) method
        - 'e': Kocifaj (1996) and Schüler (2012) method
        - 'f': Huestis(2001) method (the most accurate)

    Returns
    -------
    float or ndarray
        Chapman function value(s) representing the optical slant air mass factor 
        (ratio of slant to vertical path) for a spherical atmosphere. 
        In plane-parallel limit this reduces to 1/mu.

    References
    ----------
    .. [1] Vasylyev, D. (2021). Accurate analytic approximation for the Chapman
           grazing incidence function. Earth Planets Space, 73, 112.
           https://doi.org/10.1186/s40623-021-01435-y
    .. [2] Huestis, D. L. (2001). Accurate evaluation of the Chapman function for
           atmospheric attenuation. Journal of Quantitative Spectroscopy and Radiative
           Transfer, 69(6), 709–721. https://doi.org/10.1016/S0022-4073(00)00107-2
    .. [3] Titheridge, J. E. (1988). A non-linear theory of the Chapman graticule.
           Journal of Atmospheric and Solar-Terrestrial Physics, 50(3), 209-219.
    .. [4] Titheridge, J. E. (2000). Modelling of the basic mechanisms controlling the
           pre-reversal enhancement of equatorial spread F. Journal of Atmospheric
           and Solar-Terrestrial Physics, 62(17), 1521-1531.
    .. [5] Kocifaj, M. (1996). On the Chapman graticule computation revisited.
           Journal of Atmospheric and Solar-Terrestrial Physics, 58(16), 1819-1823.
    .. [6] Schüler, T. (2012). On the use of ionospheric scale heights in the Chapman
           function model. Radio Science, 47, RS5003.
    """

    methods_ok = ['d', 'e', 'f']

    if method not in methods_ok:
        raise ValueError(f"Invalid method '{method}'. Valid options are {methods_ok}.")
    
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

    elif method == "f":
        sin_th = np.sqrt(1.0 - mu**2)
        arg = np.sqrt(X*(1-sin_th))
        return np.sqrt((pi*X)/(1+sin_th)) * erfcx(arg)

