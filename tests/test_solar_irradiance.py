from matplotlib import pyplot as plt
from eotools.solar_irradiance import solar_irradiance_lisird
from core.conftest import savefig, pytest_runtest_makereport  # noqa


def test_solar_irradiance(request):
    SI = solar_irradiance_lisird('1nm')
    for var in ['SSI', 'SSI_UNC']:
        plt.figure()
        SI[var].plot()
        plt.title(var)
        plt.grid(True)
        savefig(request)
