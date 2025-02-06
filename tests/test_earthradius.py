#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from eotools.earth_radius import earth_radius
from eotools.earth_radius import geocentric_to_geodetic_lat
from eotools.earth_radius import geodetic_to_geocentric_lat
from eotools.earth_radius import EQUATOR_RADIUS
from eotools.earth_radius import POLES_RADIUS


def test_earthradius():
    # Verify values at EQUATOR and POLES
    assert (earth_radius(0., lat_type='geocentric') == EQUATOR_RADIUS)
    assert (earth_radius(90., lat_type='geocentric') == POLES_RADIUS)
    assert (earth_radius(-90., lat_type='geocentric') == POLES_RADIUS)
    assert (earth_radius(0., lat_type='geodetic') == EQUATOR_RADIUS)
    assert (earth_radius(90., lat_type='geodetic') == POLES_RADIUS)
    assert (earth_radius(-90., lat_type='geodetic') == POLES_RADIUS)

    # Verify with lat value between equator and poles
    geod_lat = 30.
    geoc_lat = geodetic_to_geocentric_lat(geod_lat)
    assert(np.isclose(geocentric_to_geodetic_lat(geoc_lat), geod_lat, 0., 1e-14))
    rad_geod = earth_radius(geod_lat, 'geodetic')
    rad_geoc = earth_radius(geoc_lat, 'geocentric')
    assert(np.isclose(geoc_lat, 29.833635809477595, 0., 1e-15))
    assert(np.isclose(rad_geod, 6372.824420282859, 0., 1e-15))
    assert(np.isclose(rad_geoc, 6372.824420282859, 0., 1e-15))


