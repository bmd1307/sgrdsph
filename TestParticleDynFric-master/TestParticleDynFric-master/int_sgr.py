# module for integr

from astropy.constants import G
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

import random

# helper functions

def hernquist_menc(mtot, r, c):
    return mtot * r * r / (r + c) / (r + c)

def r_tidal(sat_mass, mw_mass, mw_c, r_i):
    return r_i * (sat_mass / 3 / hernquist_menc(mw_mass, r_i, mw_c)) ** (1/3)

# main functions

def read_file(file_name,\
              columns = [0, 1, 2, 3, 4, 5],\
              pos_unit = u.kpc, \
              vel_unit = (u.km / u.s),\
              skip_lines = []):
    part_file = open(file_name)
    particle_list = []
    line_counter = 0
    for line in part_file:
        #print(line)
        
        if line_counter in skip_lines:
            line_counter = line_counter + 1
            continue
        
        if line.strip() == '' or line.strip()[0] == '#':
            line_counter = line_counter + 1
            # skip all lines beginning with this character: '#'
            continue
        particle_list.append([float(line.split()[col]) for col in columns])

        line_counter = line_counter + 1

    part_file.close()

    particle_list = np.transpose(particle_list)

    positions = \
        ([particle_list[i] for i in range(0, 3)] * pos_unit).to(u.kpc)
    velocities = \
        ([particle_list[i] for i in range(3, 6)] * vel_unit).to(u.kpc / u.Myr)

    return gd.PhaseSpacePosition(pos=positions, vel=velocities)

    

def integrate(w0,\
              sat_mass = 1e8,\
              pot=gp.HernquistPotential(m=130.0075e10*u.Msun,c = 32.089, units=galactic),\
              timestep = -10.67,\
              ntimesteps = 250,\
              verbose=False):
    
    # TODO
    orbit = None # define the orbit object in this scope

    if verbose:
        print('integrating ... ')
        start_time = time.time()
        orbit = gp.Hamiltonian(pot).integrate_orbit(w0, dt=timestep, n_steps=ntimesteps)
        orbit_time = time.time() - start_time
        print('done integrating')
        print('Length of integration: %1.3f' % orbit_time)
    else:
        orbit = gp.Hamiltonian(pot).integrate_orbit(w0, dt=timestep, n_steps=ntimesteps)
    
    return orbit, pot

def integrate_dyn_fric(w0,\
              sat_mass = 1e8,\
              pot=gp.HernquistPotential(m=130.0075e10*u.Msun,c = 32.089, units=galactic),\
              scale_radius = 18.927757889861788,\
              timestep = -10.67,\
              ntimesteps = 250,\
              verbose=False):

    def F_dyn_fric(t, w, msat, rs):
        G_gal = 4.49850215e-12 # gravitational constant in galactic units (kpc^3 Msun^-1 Myr^-2)
        accdf = 0.0
        
        q = w[0:3]
        p = w[3:6]

        verf = np.vectorize(math.erf)

        absr = np.linalg.norm(q, axis=0)
        absv = np.linalg.norm(p, axis=0)

        loglambda = np.maximum(0, np.log(absr / 3 / 1.6))
        rhoh = pot.density(q).value

        normalized_r = absr / rs

        # 0.2199 kpc / Myr = 215 km/s
        sigma = 0.2199 * ( 1.4393 * normalized_r ** 0.354 / (1 + 1.1756 * normalized_r ** 0.725) )

        chand_x = absv / math.sqrt(2) / sigma

        # k is the part of the equation with the erf and exp components
        k = verf(chand_x) - 2 * chand_x * np.exp(- np.square(chand_x)) / math.sqrt(math.pi)

        accdf = (4 * math.pi * G_gal * G_gal * msat * loglambda * rhoh / absv**3) * k * p
        
        q_dot = p
        p_dot = -pot.gradient(q).value + accdf
        

        toreturn = np.concatenate((q_dot, p_dot))
        
        return toreturn

    integrator = gi.LeapfrogIntegrator(F_dyn_fric, func_args = (sat_mass, scale_radius), func_units = galactic)

    orbit = None

    if verbose:
        print('integrating ... ')
    
        start_time = time.time()
        orbit = integrator.run(w0,dt=timestep, n_steps = ntimesteps)
        
        orbit_time = time.time() - start_time
        print('done integrating')
        print('Length of integration: %1.3f' % orbit_time)
    else:
        orbit = integrator.run(w0,dt=timestep, n_steps = ntimesteps)

    return orbit, pot

def calc_com(sat_mass, ps_vect):

    part_mass = (sat_mass / len(ps_vect.pos)) * u.Msun
    GM_value = (part_mass * part_mass * G).to(u.g * u.cm * u.cm * u.cm / u.s / u.s).value

    coords = ps_vect.pos.xyz.transpose()
    vels = ps_vect.vel.d_xyz.transpose()

    coords_cm = coords.to(u.cm).value

    boundness_list = []

    for i in range(len(coords)):

        particles_energy = 0.0 # Units cm cm / s / s

        curr_dists = np.linalg.norm(coords_cm - coords_cm[i], axis = 1)

        for j in range(len(coords)):
            if i == j:
                continue
            curr_dist_value = curr_dists[j]
            
            curr_energy = -GM_value / curr_dist_value

            particles_energy = particles_energy + curr_energy

        # the last index here is the particle ID
        boundness_list.append((particles_energy, coords[i], vels[i], i))

    boundness_list = sorted(boundness_list, key=lambda curr_tup: curr_tup[0])

    return boundness_list[0][1], boundness_list[0][2], boundness_list[0][3]


def calc_dps(sat_mass, orbit, pot, com_index, \
             m_mw = 130.0075e10 * u.Msun, \
             c_mw = 32.089 * u.kpc, \
             ylims = [0.1, 500], \
             plot_title = 'Dps of 10 particles (within 0.5 * tidal radius)',\
             show_plot=True,\
             verbose=True):
    
    if verbose:
        print(m_mw, c_mw)
        print(com_index)

    tidal_radii = []
    escape_velocities = []
    # calculate the escape velocity and tidal radius
    for ts in range(orbit.ntimes):
        curr_pos = orbit[ts].pos[com_index]
        curr_vel = orbit[ts].vel[com_index]
        r_i = np.linalg.norm(curr_pos.xyz) * u.kpc
        
        #r_tide_i = r_i * (sat_mass * u.Msun / 3 / hernquist_menc(m_mw, r_i, c_mw)) ** (1/3)
        r_tide_i = r_tidal(sat_mass * u.Msun, m_mw, c_mw, r_i)
        
        vesc_i = np.sqrt((2 * G * sat_mass * u.Msun / r_tide_i))

        tidal_radii.append(r_tide_i.to(u.kpc).value)
        escape_velocities.append(vesc_i.to(u.kpc / u.Myr).value)

    # dictionary from a timestep to a list of each particle's psvectors (ndarray)
    # calculate the normalized ps vector for each particle at each timestep
    ps_vectors = {}
    list_ps_indices = []
    num_particles = len(orbit[0].pos)

    #print('r tide', tidal_radii)

    if verbose:
        print('normalizing positions and velocities ... ')

    pos_table = orbit.pos.xyz.value
    vel_table = orbit.vel.d_xyz.value

    if verbose:
        print(pos_table.shape)


    if False:
        for part_index in range(num_particles):
            if verbose:
                if part_index == com_index:
                    continue
                if part_index % 10 == 0:
                    print('part_index', part_index)
            
            ps_vectors[part_index] = {}
            for ts in range(orbit.ntimes):
                pos_x = (pos_table[0][ts][part_index] - pos_table[0][ts][com_index]) / tidal_radii[ts]
                pos_y = (pos_table[1][ts][part_index] - pos_table[1][ts][com_index]) / tidal_radii[ts]
                pos_z = (pos_table[2][ts][part_index] - pos_table[2][ts][com_index]) / tidal_radii[ts]

                vel_x = (vel_table[0][ts][part_index] - vel_table[0][ts][com_index]) / escape_velocities[ts]
                vel_y = (vel_table[1][ts][part_index] - vel_table[1][ts][com_index]) / escape_velocities[ts]
                vel_z = (vel_table[2][ts][part_index] - vel_table[2][ts][com_index]) / escape_velocities[ts]

                ps_vectors[part_index][ts] = np.array(\
                    [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z])

    
    delta_pos = \
            orbit.pos.xyz.value - orbit.pos.xyz.value[:,:,com_index,None]
    delta_vel = \
            orbit.vel.d_xyz.value - orbit.vel.d_xyz.value[:,:,com_index,None]

    r_tidals = np.array(tidal_radii)
    v_escs = np.array(escape_velocities)

    normed_pos = delta_pos / r_tidals[:, None]
    normed_vel = delta_vel / v_escs[:, None]

    ps_vectors = np.append(normed_pos, normed_vel, axis=0).transpose()

    # find the minimum ps_vector magnitude for each particle

    # particle index to minimum ps_vector
    min_ps_vects = {}

    for part_index in range(num_particles):
        if verbose:
            if part_index == com_index:
                continue

            if part_index % 10 == 0:
                print('mix dps vect', part_index)

        curr_part = ps_vectors[part_index,:,:]

        curr_dps_mags = np.linalg.norm(curr_part, axis=1)

        # get the timestep where the dps_mag is the smallest
        min_ts = np.where(curr_dps_mags == np.min(curr_dps_mags))[0][0]

        min_ps_vects[part_index] = curr_part[min_ts]

    # append the minimum ps-vector magnitude for each particle to a matrix        
    min_matrix = np.array(list(min_ps_vects.values()))

    # find the covariance matrix
    #   transpose the matrix and call np.cov()
    cov_mat = np.cov(min_matrix.T)
    if verbose:
        print('Covariance Matrix')
        print(cov_mat)
    
    # calculate the determinant (np.linalg.det())
    toreturn = np.linalg.det(cov_mat)
    if verbose:
        print('Determinant')
        print(toreturn)
    
    # calculate the minimum magnitude of this dps vector

    if len(ps_vectors) < 10:
        part_indices = list(range(ps_vectors.shape[0]))
    else:
        part_indices = random.sample(list(range(ps_vectors.shape[0])), 10)

    # plot the dps (log on the yscale, ylim is [0.1, 50], xlim is [0, 2650] - that should be different
    # plt.hlines(2, 0, 2650, linestyles = 'dashed'

    if show_plot:
        timesteps = list(range(orbit.ntimes))
        ts_list_myr = [-2650.0 * ts / len(timesteps) for ts in timesteps]

        for part_index in part_indices:
            dps_vals = [np.linalg.norm(ps_vectors[part_index][ts]) for ts in timesteps]

            min_dps_index = dps_vals.index(min(dps_vals))
            min_dps_time = -2650.0 * min_dps_index / len(timesteps)
            curr_min_dps = min(dps_vals)

            curr_line = plt.plot(ts_list_myr, dps_vals, lw=0.5)
            print(curr_line[0].get_c())
            plt.plot([min_dps_time], [curr_min_dps], marker='+', c='black', markersize=10.0, mew=1.5, zorder = 10)

        plt.hlines(2, min(ts_list_myr), max(ts_list_myr), linestyles = 'dashed')

        plt.title(plot_title)
        plt.ylabel('Dps magnitude')
        plt.xlabel('Time (Myr, 0 = present)')
        plt.yscale('log')
        plt.ylim(ylims)
        plt.xlim([min(ts_list_myr), max(ts_list_myr)])
        plt.show()

    return toreturn

def orbit_video(orb, folder_name, axes=['x', 'y']):
    save_counter = 10000

    orb_arr = [o for o in orb]
    timesteps = orb.t.value.tolist() # These are Myr
    
    orb_arr.reverse()
    timesteps.reverse()
    
    
    for i in range(len(orb_arr)):
        curr_ts = timesteps[i]
        o = orb_arr[i]
        
        curr_fig = o.plot(axes, s = 2.0, edgecolors='none', color = 'black')
        curr_fig.set_size_inches(6.0, 5.0)

        plt.title('Sgr dSph orbit (t=%1.2f Myr)' % curr_ts)

        plt.xlabel('X (kpc)')
        plt.ylabel('Y (kpc')
        
        plt.xlim([-120, 120])
        plt.ylim([-120, 120])

        plt.gca().set_aspect('equal', adjustable='box')
        
        curr_fig.savefig(\
            'gaiavideos\\' + folder_name + '\\p' + str(save_counter)[1:] + '.png')
        curr_fig.clear()
        plt.close(curr_fig)
        save_counter = save_counter + 1
        print((save_counter - 10000), end = ' ')
    print('Saved images')

def plot_bound_parts():
    # TODO
    pass

def select_tidal_annulus(w0,\
                    ann_low = 0.0,\
                    ann_high = 0.5,\
                    sat_mass=1e8,\
                    mw_mass=130.0075e10,\
                    mw_c=32.089):
    com_p, com_v, com_index = calc_com(sat_mass, w0)


    distances_to_com =\
        np.linalg.norm((w0.pos - CartesianRepresentation(com_p)).xyz, axis=0)

    tidal_radius = r_tidal(sat_mass, mw_mass, mw_c, np.linalg.norm(com_p))

    selected_indices = []

    for p_index in range(len(distances_to_com)):
        curr_dist = distances_to_com[p_index]
        if (ann_low * tidal_radius) <= curr_dist < (ann_high * tidal_radius):
            selected_indices.append(p_index)

    return w0[selected_indices]



