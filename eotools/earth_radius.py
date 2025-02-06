#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

EQUATOR_RADIUS = 6378.137   # in kilometers
POLES_RADIUS = 6356.7523142 # in kilometers


def earth_radius(lat, lat_type='geocentric'):
    """
    Compute the earth radius giving the geocentric or geodetic latitude

    Parameters
    ----------
    lat : float,
        Geocentric or geodetic latitude in degrees
    lat_type : str, optional
        The latitude type, 2 options: "geocentric", "geodetic"
    
    Returns
    -------
    rad : float
        The earth radius at the given latitude
    """
    a = EQUATOR_RADIUS
    b = POLES_RADIUS

    if lat_type == 'geocentric':
        a2 = a*a
        b2 = b*b
        coslat = math.cos(math.radians(lat))
        coslat2 = coslat*coslat
        rad = b / (math.sqrt(1 - (1-(b2/a2))*coslat2))
    elif lat_type == 'geodetic':
        coslat = math.cos(math.radians(lat))
        sinlat = math.sin(math.radians(lat))
        rad = math.sqrt(  ( (a**2 * coslat)**2 +  (b**2 * sinlat)**2  ) /
                    ( (a * coslat)**2 +  (b * sinlat)**2 )  )
    else:
        raise ValueError("The value of parameter lat_type must be: 'geocentric' or 'geodetic'")

    return rad


def geocentric_to_geodetic_lat(lat):
    """
    Convert geocentric latitude to geodetic latitude

    Parameters
    ----------
    lat : float
        Geocentric latitude in degrees
    
    Returns
    -------
    out : float
        Geodetic latitude in degrees
    """
    a = EQUATOR_RADIUS
    b = POLES_RADIUS
    a2 = a*a
    b2 = b*b
    return math.degrees(math.atan((a2/b2)*math.tan(math.radians(lat))))


def geodetic_to_geocentric_lat(lat):
    """
    Convert geodetic latitude to geocentric latitude

    Parameters
    ----------
    lat : float
        Geodetic latitude in degrees
    
    Returns
    -------
    out : float
        Geocentric latitude in degrees
    """
    a = EQUATOR_RADIUS
    b = POLES_RADIUS
    a2 = a*a
    b2 = b*b
    return math.degrees(math.atan((b2/a2)*math.tan(math.radians(lat))))