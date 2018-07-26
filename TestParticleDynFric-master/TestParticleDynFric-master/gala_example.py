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

def vary_potential():

    sag_mass = 1e8
    
    normal_mw_mass = 130.0075e10*u.Msun
    normal_mw_conc = 9.39
    normal_mw_rs = 18.927757889861788 * u.kpc

    w0 = int_sgr.read_file('centroid_part_1000', columns=[3, 4, 5, 6, 7, 8], skip_lines=[0])

    gvs = []

    m_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    c_ratios =  [0.8, 0.9, 1.0, 1.1, 1.2]
    rs_ratios =  [0.8, 0.9, 1.0, 1.1, 1.2]

    for m_ratio in m_ratios:
        for c_ratio in c_ratios:
            for rs_ratio in rs_ratios:
                curr_mass = normal_mw_mass * m_ratio
                curr_c = normal_mw_conc * c_ratio
                curr_rs = normal_mw_rs * rs_ratio
                print('Theta:',\
                      'mass =', curr_mass,\
                      'c =', curr_c,\
                      'rs =', curr_rs,\
                      )
                curr_pot = int_sgr.get_potential(total_mass=curr_mass, r_scale=curr_rs, mw_conc=curr_c)
                orbit, pot = int_sgr.integrate(w0, pot=curr_pot)

                com_p, com_v, com_index = int_sgr.calc_com(w0)
                com_orbit,pot = int_sgr.integrate(gd.PhaseSpacePosition(pos=com_p, vel=com_v))

                gv = int_sgr.calc_dps(sag_mass, orbit, curr_pot, com_orbit)
                print('Generalized Variance:', gv)

                gvs.append(gv)
    np.set_printoptions(precision=3)
    print(np.reshape(gvs, [len(m_ratios), len(c_ratios), len(rs_ratios)]))

def animate_orbit():
    w0 = int_sgr.read_file('heavy_sag_7_10', columns=[3, 4, 5, 6, 7, 8], skip_lines=[0])

    curr_pot = int_sgr.get_potential()
    orbit, pot = int_sgr.integrate(w0, pot=curr_pot)

    int_sgr.orbit_video(orbit, 'heavy sag animation')

def select_particles():
    w0 = int_sgr.read_file('centroid_part_1000', columns=[3, 4, 5, 6, 7, 8], skip_lines=[0])

    curr_pot = int_sgr.get_potential()
    orbit, pot = int_sgr.integrate(w0, pot=curr_pot)

    fig = orbit[0].plot()
    fig.get_axes()[1].set_title('Full Sgr Stream')
    fig.show()

    w_half_tidal = int_sgr.select_tidal_annulus(\
                    w0, ann_low = 0.0, ann_high = 0.5,\
                    sat_mass=1e8, mw_mass=130.0075e10, mw_c=32.089)

    tidal_orbit, pot = int_sgr.integrate(w_half_tidal, pot=curr_pot)

    fig = tidal_orbit[0].plot()
    fig.get_axes()[1].set_title('Sgr Stream r < 0.5 r_tidal')
    fig.show()

    

def __main__():
    #integrate_normal_orbit()
    #integrate_dyn_fric()
    #plot_dps()
    #vary_potential()

    #animate_orbit()

    select_particles()

__main__()
