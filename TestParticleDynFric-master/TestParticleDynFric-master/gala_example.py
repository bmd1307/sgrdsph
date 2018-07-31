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

# shows the format of a simple orbit integration, and plots the orbit at the first and last timestep
def integrate_normal_orbit():
    # reads the coordinates from the file to w0
    w0 = int_sgr.read_file('centroid_part_1000', columns=[3, 4, 5, 6, 7, 8], skip_lines=[0])

    # integrates the orbits with initial conditions w0 and with verbose mode on
    orbit, pot = int_sgr.integrate(w0, verbose=True)

    # plots the orbit at the first timestep (2.65 Gyr ago)
    fig = orbit[-1].plot()
    # adds a title to the default plotting function from gala
    fig.get_axes()[1].set_title('Sgr Stream (2.65 Gyr ago)')
    fig.show()

    # plots the orbit at the present timestep
    fig = orbit[0].plot()
    fig.get_axes()[1].set_title('Sgr Stream (Present)')
    fig.show()
    
# performs a basic integration with dynamical friction, and plots the orbit at the first and last timestep
def integrate_dyn_fric():
    sat_mass = 1e10 # Msun

    # reads the coordinates from the file to w0
    w0 = int_sgr.read_file('heavy_sag_7_10', columns=[3, 4, 5, 6, 7, 8], skip_lines=[0])

    # integrates the orbits with initial conditions w0, dynamical friction and with verbose mode on
    orbit, pot = int_sgr.integrate_dyn_fric(w0, sat_mass = sat_mass, verbose=True)

    # plots the orbit at the first timestep (2.65 Gyr ago)
    fig = orbit[-1].plot()
    fig.get_axes()[1].set_title('Sgr Stream (2.65 Gyr ago)')
    fig.show()

    # plots the orbit at the present timestep
    fig = orbit[0].plot()
    fig.get_axes()[1].set_title('Sgr Stream (Present)')
    fig.show()

# demonstrates the plotting feature of the calc_dps function
def plot_dps():
    # the parameters for the Sgr dSph
    sag_mass = 1e8
    
    # reads the coordinates from the file to w0
    w0 = int_sgr.read_file('centroid_part_1000', columns=[3, 4, 5, 6, 7, 8], skip_lines=[0])

    # integrates the orbits with initial conditions w0
    orbit, pot = int_sgr.integrate(w0)

    # finds the center of mass of the particle distribution and integrates its orbit
    com_p, com_v, com_index = int_sgr.calc_com(w0)
    com_orbit,pot = int_sgr.integrate(gd.PhaseSpacePosition(pos=com_p, vel=com_v))

    # calculates the generalized variance of the particle distribution in verbose mode, and displays the dps plot
    gv = int_sgr.calc_dps(sag_mass, orbit, pot, com_orbit, \
                          ylims = [0.2, 400], plot_title = 'Example Dps Plot', show_plot=True, verbose=True)
    print('Generalized Variance:', gv)

# Calculates the generalized variance while varying the potential parameters
def vary_potential():

    # the normal parameters for the potential and the Sgr dSph
    sag_mass = 1e8
    
    normal_mw_mass = 130.0075e10*u.Msun
    normal_mw_conc = 9.39
    normal_mw_rs = 18.927757889861788 * u.kpc

    # reads the coordinates from the file to w0
    w0 = int_sgr.read_file('centroid_part_1000', columns=[3, 4, 5, 6, 7, 8], skip_lines=[0])

    gvs = []

    m_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    c_ratios =  [0.8, 0.9, 1.0, 1.1, 1.2]
    rs_ratios =  [0.8, 0.9, 1.0, 1.1, 1.2]

    # makes a potential using 80%, 90%, 100%, 110% and 120% of each of the normal potential parameters
    for m_ratio in m_ratios:
        for c_ratio in c_ratios:
            for rs_ratio in rs_ratios:
                # gets the current potential parameters
                curr_mass = normal_mw_mass * m_ratio
                curr_c = normal_mw_conc * c_ratio
                curr_rs = normal_mw_rs * rs_ratio
                print('Theta:',\
                      'mass =', curr_mass,\
                      'c =', curr_c,\
                      'rs =', curr_rs,\
                      )

                # creates the current potential object and uses it to integrate the particle orbits
                curr_pot = int_sgr.get_potential(total_mass=curr_mass, r_scale=curr_rs, mw_conc=curr_c)
                orbit, pot = int_sgr.integrate(w0, pot=curr_pot)

                # finds the center of mass of the particle distribution and integrates its orbit
                com_p, com_v, com_index = int_sgr.calc_com(w0)
                com_orbit,pot = int_sgr.integrate(gd.PhaseSpacePosition(pos=com_p, vel=com_v))

                # calculates the generalized variance of the particle distribution
                gv = int_sgr.calc_dps(sag_mass, orbit, curr_pot, com_orbit)
                print('Generalized Variance:', gv)

                gvs.append(gv)

    # prints the generalized variances calculated from the potential parameters
    np.set_printoptions(precision=3)
    print(np.reshape(gvs, [len(m_ratios), len(c_ratios), len(rs_ratios)]))


# produces and saves the orbit images for an orbit integration to the folder 'heavy sag animation'
def animate_orbit():
    # reads the coordinates from the file to w0
    w0 = int_sgr.read_file('heavy_sag_7_10', columns=[3, 4, 5, 6, 7, 8], skip_lines=[0])

    # gets the default potential object and integrates the orbits using the initial conditions in w0
    curr_pot = int_sgr.get_potential()
    orbit, pot = int_sgr.integrate(w0, pot=curr_pot)

    # plots and saves the particle positions at each timestep to a new folder: 'heavy sag animation'
    # NOTE: the folder must not already exist, otherwise an error will be raised
    int_sgr.orbit_video(orbit, 'heavy sag animation')


# selects particles within 0.5 r_tidal, and plots them along with the total stream
def select_particles():
    # reads the coordinates from the file to w0
    w0 = int_sgr.read_file('centroid_part_1000', columns=[3, 4, 5, 6, 7, 8], skip_lines=[0])

    # gets the default potential object and integrates the orbits using the initial conditions in w0
    curr_pot = int_sgr.get_potential()
    orbit, pot = int_sgr.integrate(w0, pot=curr_pot)

    # plots the orbit at the present timestep
    fig = orbit[0].plot()
    fig.get_axes()[1].set_title('Full Sgr Stream')
    fig.show()

    # selects the particles within 0.5 * r_tidal and saves them to w_half_tidal
    w_half_tidal = int_sgr.select_tidal_annulus(\
                    w0, ann_low = 0.0, ann_high = 0.5,\
                    sat_mass=1e8, mw_mass=130.0075e10, mw_c=32.089)

    # integrates the orbits of the particles within 0.5 r_tidal
    tidal_orbit, pot = int_sgr.integrate(w_half_tidal, pot=curr_pot)

    # plots the orbit at the present timestep
    fig = tidal_orbit[0].plot()
    fig.get_axes()[1].set_title('Sgr Stream r < 0.5 r_tidal')
    fig.show()

# selects particles within 0.5 r_tidal, and saves their coordinates to a text file
def test_export(file_name):
    sat_mass = 1e8 * u.Msun
    # reads the coordinates from the file to w0
    w0 = int_sgr.read_file('centroid_part_1000', columns=[3, 4, 5, 6, 7, 8], skip_lines=[0])

    # selects the particles within 0.5 * r_tidal and saves them to w_half_tidal
    w_half_tidal = int_sgr.select_tidal_annulus(\
                    w0, ann_low = 0.0, ann_high = 0.5,\
                    sat_mass=1e8, mw_mass=130.0075e10, mw_c=32.089)

    # writes the particle data for the particles within 0.5 * r_tidal to the file specified by 'file_name'
    int_sgr.export(w_half_tidal.pos, w_half_tidal.vel, sat_mass, file_name)

    print('%s has been written' % file_name)

    

def __main__():
    # Uncomment the function you would like to run
    
    integrate_normal_orbit()
    #integrate_dyn_fric()
    #plot_dps()
    #vary_potential()
    #animate_orbit()
    #select_particles()
    #test_export('sag_in_half_tidal')

__main__()
