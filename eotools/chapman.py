import numpy as np
from scipy.special import erfc


def chapman(mu, X=6371. / 8.0, method="d"):
    """
    Compute the Chapman function for atmospheric absorption calculations.

    The Chapman function represents the ratio of the slant path to the vertical path
    through a stratified atmosphere. Two methods are available for computation, each
    with different theoretical bases.

    Parameters
    ----------
    mu : float or array_like
        Cosine of the zenith angle at the point of interest.
    X : float, optional
        Ratio of altitude at the point of interest (from Earth's centre) to the
        scale height of the atmospheric absorber. Default is 6371/8 ≈ 796.375,
        corresponding to Earth's mean radius (6371 km) and Rayleigh scale height (8 km).
    method : {'d', 'e'}, optional
        Method for computation. Default is 'd'.
        
        - 'd': Titheridge (1988, 2000) method
        - 'e': Kocifaj (1996) and Schüler (2012) method

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
    .. [2] Titheridge, J. E. (1988). A non-linear theory of the Chapman graticule.
           Journal of Atmospheric and Solar-Terrestrial Physics, 50(3), 209-219.
    .. [3] Titheridge, J. E. (2000). Modelling of the basic mechanisms controlling the
           pre-reversal enhancement of equatorial spread F. Journal of Atmospheric
           and Solar-Terrestrial Physics, 62(17), 1521-1531.
    .. [4] Kocifaj, M. (1996). On the Chapman graticule computation revisited.
           Journal of Atmospheric and Solar-Terrestrial Physics, 58(16), 1819-1823.
    .. [5] Schüler, T. (2012). On the use of ionospheric scale heights in the Chapman
           function model. Radio Science, 47, RS5003.
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