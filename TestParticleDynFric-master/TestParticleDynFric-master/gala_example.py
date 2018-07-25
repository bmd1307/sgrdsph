# example functions for gala integration

from astropy.coordinates import CartesianRepresentation
from astropy.coordinates import CartesianDifferential
import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math

import numpy as np

import time

import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp

from gala.units import galactic

import int_sgr

def integrate_normal_orbit():
    sat_mass = 1e8 # Msun

    mw_conc = 9.39
    mw_r_scale = 18.927757889861788 * u.kpc

    mw_mass = 130.0075e10*u.Msun
    #mw_c = 32.089 * u.kpc
    a_term = math.sqrt(2 * (math.log(1 + mw_conc) - mw_conc / (1+mw_conc)))
    mw_c = mw_r_scale * a_term

    print(a_term)
    print('mw_c', mw_c)

    w0 = int_sgr.read_file('centroid_part_1000', columns=[3, 4, 5, 6, 7, 8], skip_lines=[0])

    #pot = gp.HernquistPotential(m=mw_mass,c =mw_c, units=galactic)
    
    orbit, pot = int_sgr.integrate(w0)

    orbit[-1].plot().show()
    orbit[0].plot().show()

def integrate_dyn_fric():
    sat_mass = 1e10 # Msun

    mw_conc = 9.39
    mw_r_scale = 18.927757889861788 * u.kpc

    mw_mass = 130.0075e10*u.Msun
    #mw_c = 32.089 * u.kpc
    a_term = math.sqrt(2 * (math.log(1 + mw_conc) - mw_conc / (1+mw_conc)))
    mw_c = mw_r_scale * a_term

    print(a_term)
    print('mw_c', mw_c)

    w0 = int_sgr.read_file('heavy_sag_7_10', columns=[3, 4, 5, 6, 7, 8], skip_lines=[0])

    #pot = gp.HernquistPotential(m=mw_mass,c =mw_c, units=galactic)
    
    orbit, pot = int_sgr.integrate_dyn_fric(w0, sat_mass = 1e8)

    orbit[-1].plot().show()
    #orbit[0].plot().show()

def plot_dps():
    pass

def __main__():
    #integrate_normal_orbit()
    integrate_dyn_fric()
    #plot_dps()

__main__()
