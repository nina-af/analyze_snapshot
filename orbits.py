import numpy as np

from numpy import arccos as acos
from numpy import cos, dot, sin, sqrt
from numpy.linalg import norm
from scipy.constants import pi

# Unit conversions.
cm_to_au = 6.6845871226706e-14
pc_to_m = 3.08567758128e16
pc_to_au = 206264.80624538
m_to_au = 6.6845871226706e-12


# Units for orbital elements calculations
# [r] = m
# [v] = m/s
# [mu] = m * (m/s)^2

def angular_momentum(position, velocity):
    return np.cross(position, velocity)

def node_vector(angular_momentum):
    return np.cross([0, 0, 1], angular_momentum)

def eccentricity_vector(position, velocity, mu):
    r = position
    v = velocity
    ev = 1.0 / mu * ((norm(v) ** 2 - mu / norm(r)) * r - dot(r, v) * v)
    return ev

def specific_orbital_energy(position, velocity, mu):
    r = position
    v = velocity
    return norm(v) ** 2 / 2 - mu / norm(r)

def elements_from_state_vector(r, v, mu):
    h = angular_momentum(r, v)
    n = node_vector(h)

    ev = eccentricity_vector(r, v, mu)

    E = specific_orbital_energy(r, v, mu)

    a = -mu / (2.0 * E)
    e = norm(ev)

    SMALL_NUMBER = 1e-15

    # Inclination is the angle between the angular
    # momentum vector and its z component.
    i = acos(h[2] / norm(h))

    if abs(i - 0) < SMALL_NUMBER:
        # For non-inclined orbits, raan is undefined;
        # set to zero by convention
        raan = 0
        if abs(e - 0) < SMALL_NUMBER:
            # For circular orbits, place periapsis
            # at ascending node by convention
            arg_pe = 0
        else:
            # Argument of periapsis is the angle between
            # eccentricity vector and its x component.
            arg_pe = acos(ev[0] / norm(ev))
    else:
        # Right ascension of ascending node is the angle
        # between the node vector and its x component.
        raan = acos(n[0] / norm(n))
        if n[1] < 0:
            raan = 2 * pi - raan

        # Argument of periapsis is angle between
        # node and eccentricity vectors.
        arg_pe = acos(dot(n, ev) / (norm(n) * norm(ev)))

    if abs(e - 0) < SMALL_NUMBER:
        if abs(i - 0) < SMALL_NUMBER:
            # True anomaly is angle between position
            # vector and its x component.
            f = acos(r[0] / norm(r))
            if v[0] > 0:
                f = 2 * pi - f
        else:
            # True anomaly is angle between node
            # vector and position vector.
            f = acos(dot(n, r) / (norm(n) * norm(r)))
            if dot(n, v) > 0:
                f = 2 * pi - f
    else:
        if ev[2] < 0:
            arg_pe = 2 * pi - arg_pe

        # True anomaly is angle between eccentricity
        # vector and position vector.
        f = acos(dot(ev, r) / (norm(ev) * norm(r)))

        if dot(r, v) < 0:
            f = 2 * pi - f
            
    return a, e, i, raan, arg_pe, f

