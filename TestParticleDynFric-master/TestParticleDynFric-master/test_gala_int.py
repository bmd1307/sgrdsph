# gala test

print('importing ...')

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

print('done')

def read_particles(file_name = 'centroid_part_1000'):
    part_file = open(file_name)
    next(part_file)
    particle_list = []
    for line in part_file:
        curr_vals = [float(val) for val in line.split()]
        if len(curr_vals) == 9:
            particle_list.append(curr_vals[3:9])

    part_file.close()

    return particle_list

def orbit_video(orb):
    save_counter = 10000

    orb_arr = [o for o in orb]
    timesteps = orb.t.value.tolist() # These are Myr
    
    orb_arr.reverse()
    timesteps.reverse()
    
    
    for i in range(len(orb_arr)):
        curr_ts = timesteps[i]
        o = orb_arr[i]
        
        curr_fig = o.plot(['x', 'y'], s = 2.0, edgecolors='none', color = 'black')
        curr_fig.set_size_inches(6.0, 5.0)

        plt.title('Sgr dSph orbit (t=%1.2f Myr)' % curr_ts)

        plt.xlabel('X (kpc)')
        plt.ylabel('Y (kpc')
        
        plt.xlim([-120, 120])
        plt.ylim([-120, 120])

        plt.gca().set_aspect('equal', adjustable='box')
        
        curr_fig.savefig(\
            'gaiavideos\\test_right_params\\p' + str(save_counter)[1:] + '.png')
        curr_fig.clear()
        plt.close(curr_fig)
        save_counter = save_counter + 1
        print((save_counter - 10000), end = ' ')
    print('Saved images')

def plot_pot(potential):

    pot_file = open('potn.txt')

    counter = 1
    lines = []
    for line in pot_file:
        if line[0] == '#':
            continue
        rad, ben_phi = [float(val) for val in line.split()]
        pot_phi = potential.acceleration(\
            [rad, 0, 0] * u.kpc).to(u.cm / u.s / u.s).value[0][0]
        lines.append("%15.5e %15.5e %15.5e %15.5e" %\
                     (rad, ben_phi, pot_phi, ben_phi / pot_phi))
        if counter % 100 == 0:
            print('.', end = '')
        counter = counter + 1            
    
    pot_file.close()

    for i in range(0, 50):
        print(lines[i*100])

def calc_com(part_mass, orbit, pot):

    part_mass = part_mass * u.Msun
    GM_value = (part_mass * part_mass * G).to(u.g * u.cm * u.cm * u.cm / u.s / u.s).value

    coords = orbit[0].pos
    vels = orbit[0].vel

    coords_values = np.array(coords.get_xyz().to(u.cm)).T.tolist()

    boundness_list = []

    for i in range(len(coords)):
        curr_p = coords[i]
        curr_v = vels[i]
        pot_energy = pot.energy(curr_p.get_xyz()).to(u.cm * u.cm / u.s / u.s) *\
                         part_mass.to(u.g)

        particles_energy = 0.0 # Units cm cm / s / s

        for j in range(len(coords)):
            if i == j:
                continue
            ix, iy, iz = coords_values[i]
            jx, jy, jz = coords_values[j]
            
            dispx = jx - ix
            dispy = jy - iy
            dispz = jz - iz

            curr_dist_value = (dispx * dispx + dispy * dispy + dispz * dispz)**.5 
            
            curr_energy = -GM_value / curr_dist_value

            particles_energy = particles_energy + curr_energy

        # the last index here is the particle ID
        boundness_list.append((particles_energy, coords[i], vels[i], i))

    boundness_list = sorted(boundness_list, key=lambda curr_tup: curr_tup[0])

    bound_parts = [curr_tup[1] for curr_tup in boundness_list[:100]]
    unbound_parts = [curr_tup[1] for curr_tup in boundness_list[100:]]

    bound_vels = [curr_tup[2] for curr_tup in boundness_list[:100]]
    unbound_vels = [curr_tup[2] for curr_tup in boundness_list[100:]]

    get_x = lambda coordinate_list: [curr_coord.x.value for curr_coord in coordinate_list]
    get_y = lambda coordinate_list: [curr_coord.y.value for curr_coord in coordinate_list]
    get_z = lambda coordinate_list: [curr_coord.z.value for curr_coord in coordinate_list]

    get_vx = lambda coordinate_list: [curr_coord.d_x.to(u.km / u.s).value for curr_coord in coordinate_list]
    get_vy = lambda coordinate_list: [curr_coord.d_y.to(u.km / u.s).value for curr_coord in coordinate_list]
    get_vz = lambda coordinate_list: [curr_coord.d_z.to(u.km / u.s).value for curr_coord in coordinate_list]

    bound_xs = get_x(bound_parts)
    bound_ys = get_y(bound_parts)
    bound_zs = get_z(bound_parts)

    bound_vxs = get_vx(bound_vels)
    bound_vys = get_vy(bound_vels)
    bound_vzs = get_vz(bound_vels)

    #print('Naive com: (%f, %f, %f)' % (np.average(bound_xs), np.average(bound_ys), np.average(bound_zs)))
    #print('Naive com vel: (%f, %f, %f)' % (np.average(bound_vxs), np.average(bound_vys), np.average(bound_vzs)))

    most_bound_part = boundness_list[0][1]

    #print(boundness_list[0][1])
    #print(boundness_list[0][2].d_xyz.to(u.km / u.s))
    #print(boundness_list[0][3])

    return boundness_list[0][1], boundness_list[0][2], boundness_list[0][3]

    #return CartesianRepresentation(np.average(bound_xs), np.average(bound_ys), np.average(bound_zs)),\
    #       CartesianDifferential((np.average(bound_vxs), np.average(bound_vys), np.average(bound_vzs)))

def calc_dps(part_mass, orbit, pot, com_index):

    m_mw = 130.0075e10 * u.Msun
    c_mw = 32.089 * u.kpc

    #com_p, com_v, com_index = calc_com(part_mass, orbit, pot)
    #com_index = 296
    #print(com_p)
    #print(com_v)
    print(com_index)

    tidal_radii = []
    escape_velocities = []
    # calculate the escape velocity and tidal radius
    for ts in range(orbit.ntimes):
        curr_pos = orbit[ts].pos[com_index]
        curr_vel = orbit[ts].vel[com_index]
        r_i = np.linalg.norm(curr_pos.xyz) * u.kpc
        #print(ts, r_i)
        #print('M enc', hernquist_menc(m_mw, r_i, c_mw))
        r_tide_i = r_i * (1e8 * u.Msun / 3 / hernquist_menc(m_mw, r_i, c_mw)) ** (1/3)
        #print('tidal radius', r_tide_i)
        vesc_i = np.sqrt((2 * G * 1e8 * u.Msun / r_tide_i))
        #print('Escape velocity', vesc_i.to(u.km / u.s))
        #print()

        tidal_radii.append(r_tide_i.to(u.kpc).value)
        escape_velocities.append(vesc_i.to(u.km / u.s).value)

    # dictionary from a timestep to a list of each particle's psvectors (ndarray)
    # calculate the normalized ps vector for each particle at each timestep
    ps_vectors = {}
    list_ps_indices = []
    num_particles = len(orbit[0].pos)

    print('normalizing positions and velocities ... ')

    pos_table = orbit.pos.xyz.value
    vel_table = orbit.vel.d_xyz.value

    print(pos_table.shape)
    print(type(pos_table))
    print(type(vel_table))

    print(type(pos_table[0][0][0]))

    #for part_index in range(num_particles):
    for part_index in range(num_particles):
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

    #print(ps_vectors[0][0])
    #print(ps_vectors[0][1])

    # find the minimum ps_vector magnitude for each particle

    # particle index to minimum ps_vector
    min_ps_vects = {}

    for part_index in range(num_particles):
        if part_index == com_index:
            continue

        if part_index % 10 == 0:
            print('mix dps vect', part_index)

        curr_part = ps_vectors[part_index]

        curr_min_ps = None
        curr_min_mag = 1e99
        
        for ts in range(orbit.ntimes):
            curr_ps_mag = np.linalg.norm(ps_vectors[part_index][ts])
            if curr_ps_mag < curr_min_mag:
                curr_min_mag = curr_ps_mag
                curr_min_ps = ps_vectors[part_index][ts]

        min_ps_vects[part_index] = curr_min_ps

    # append the minimum ps-vector magnitude for each particle to a matrix        
    min_matrix = np.array(list(min_ps_vects.values()))
    print(min_matrix[:10])

    # find the covariance matrix
    #   transpose the matrix and call np.cov()
    cov_mat = np.cov(min_matrix.T)
    print('Covariance Matrix')
    print(cov_mat)
    
    # calculate the determinant (np.linalg.det())
    print('Determinant')
    print(np.linalg.det(cov_mat))
    
    # calculate the minimum magnitude of this dps vector

    if len(ps_vectors) < 10:
        part_indices = ps_vectors.keys()
    else:
        part_indices = random.sample(ps_vectors.keys(), 10)

    # plot the dps (log on the yscale, ylim is [0.1, 50], xlim is [0, 2650] - that should be different
    # plt.hlines(2, 0, 2650, linestyles = 'dashed'

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

    plt.title('Dps of 10 particles (within 0.5 * tidal radius)')
    plt.ylabel('Dps magnitude')
    plt.xlabel('Time (Myr, 0 = present)')
    plt.yscale('log')
    plt.ylim([0.1, 500])
    plt.xlim([min(ts_list_myr), max(ts_list_myr)])
    plt.show()
        

def integrate(input_file = 'heavy_sag_core', sat_mass = 1e8):
    pot = gp.HernquistPotential(m=130.0075e10*u.Msun,c = 32.089, units=galactic)
    #pot = gp.NFWPotential(m=130e10*u.Msun,r_s = 18.9*u.kpc, units=galactic)
    #pot = gp.MilkyWayPotential()

    particle_list = read_particles(input_file)

    num_particles = len(particle_list)

    particle_list = np.transpose(particle_list).tolist()

    #print(particle_list[0] * u.kpc)

    positions = [particle_list[i] for i in range(0, 3)] * u.kpc
    velocities = [particle_list[i] for i in range(3, 6)] * (u.km / u.s)

    #positions = [[10, -10],[0,0],[0,0]] * u.kpc
    #velocities = [[0, 0], [-100, 100], [0,0]] * (u.km / u.s)

    w0 = gd.PhaseSpacePosition(pos=positions, vel=velocities)


    print('integrating ... ')
    start_time = time.time()
    orbit = gp.Hamiltonian(pot).integrate_orbit(w0,dt=-10.67, n_steps = 250)
    orbit_time = time.time() - start_time
    print('done integrating')
    print('Length of integration: %1.3f' % orbit_time)
    
    #orbit[0].plot(['x', 'y'], s = 4.0, edgecolors='none', color = 'black').show()
    xs = orbit[0].pos[:10]
    print(xs)
    print(type(xs))

    #plt.hist2d(xs, ys, bins = 30)
    #plt.show()
    #orbit_video(orbit)
    #plot_bound_parts(1e8 / 1000, orbit[-1].pos, orbit[-1].vel, pot)
    com_p, com_v, com_index = calc_com(sat_mass / num_particles, orbit, pot)
    #print(com_p, com_v, com_index)
    #com_index = 296
    calc_dps(sat_mass / num_particles, orbit, pot, com_index)
    #print(orbit.t)

def plot_bound_parts(part_mass, coords, vels, pot):
    bound_parts = []
    unbound_parts = []

    part_mass = part_mass * u.Msun
    GM_value = (part_mass * part_mass * G).to(u.g * u.cm * u.cm * u.cm / u.s / u.s).value

    print(GM_value, 'g * cm^2 / s^2')

    coords_values = np.array(coords.get_xyz().to(u.cm)).T.tolist()

    boundness_list = []

    for i in range(len(coords)):
        curr_p = coords[i]
        curr_v = vels[i]
        pot_energy = pot.energy(curr_p.get_xyz()).to(u.cm * u.cm / u.s / u.s) *\
                         part_mass.to(u.g)
        #print('Particle %i energy:\t' % i, pot_energy)

        particles_energy = 0.0 # Units cm cm / s / s

        for j in range(len(coords)):
            if i == j:
                continue
            ix, iy, iz = coords_values[i]
            jx, jy, jz = coords_values[j]
            
            dispx = jx - ix
            dispy = jy - iy
            dispz = jz - iz

            curr_dist_value = (dispx * dispx + dispy * dispy + dispz * dispz)**.5 
            
            curr_energy = -GM_value / curr_dist_value
            
            #print('\tDistance from particle %i:\t' % j, curr_dist)
            #print('\tEnergy from particle %i:\t' % j, curr_energy)

            particles_energy = particles_energy + curr_energy

        #print('Particle %i energy:\t' % i, particles_energy)
        if i % 10 == 0:
            print(i // 10)
        boundness_list.append((particles_energy, coords[i]))

        #part_i_energy = 0.5 * part_mass * np.linalg.norm(vels[i].xyz.value)

    boundness_list = sorted(boundness_list, key=lambda curr_tup: curr_tup[0])

    bound_parts = [curr_tup[1] for curr_tup in boundness_list[:100]]
    unbound_parts = [curr_tup[1] for curr_tup in boundness_list[100:]]

    bound_energy = sum([curr_tup[0] for curr_tup in boundness_list[:100]])
    tot_energy = sum([curr_tup[0] for curr_tup in boundness_list])

    print('Energy of bound particles: %e' % bound_energy)
    print('Total energy of Sgr: %e' % tot_energy)
    print('Ratio of bound energy: %f' % (bound_energy / tot_energy))

    print(bound_parts[:10])

    get_x = lambda coordinate_list: [curr_coord.x.value for curr_coord in coordinate_list]
    get_y = lambda coordinate_list: [curr_coord.y.value for curr_coord in coordinate_list]
    get_z = lambda coordinate_list: [curr_coord.z.value for curr_coord in coordinate_list]

    bound_xs = get_x(bound_parts)
    bound_ys = get_y(bound_parts)
    bound_zs = get_z(bound_parts)

    print('x range: (%f, %f)' % (min(bound_xs), max(bound_xs)))
    print('y range: (%f, %f)' % (min(bound_ys), max(bound_ys)))
    print('z range: (%f, %f)' % (min(bound_zs), max(bound_zs)))

    low_left_corner = (min(bound_xs),min(bound_ys))
    core_box_width = max(bound_xs) - min(bound_xs)
    core_box_height = max(bound_ys) - min(bound_ys)

    print('Naive com: (%f, %f, %f)' % (np.average(bound_xs), np.average(bound_ys), np.average(bound_zs)))

    unbound_xs = get_x(unbound_parts)
    unbound_ys = get_y(unbound_parts)

    plt.gca().patch.set_facecolor('white')

    plt.scatter(bound_xs, bound_ys, s = 2.0, edgecolors='none', color = 'red', zorder=10)
    plt.scatter(unbound_xs, unbound_ys, s = 2.0, edgecolors='none', color = 'black', zorder=1)

    rect = patches.Rectangle(low_left_corner, core_box_width, core_box_height, linewidth=1, edgecolor='blue', facecolor='none', zorder = 20)

    plt.gca().add_patch(rect)

    plt.title('Sgr Stream (t = present)')

    plt.xlabel('X (kpc)')
    plt.ylabel('Y (kpc)')

    plt.xlim([-60, 60])
    plt.ylim([-60, 60])

    plt.axis('equal')
    
    plt.show()

    most_bound_part = boundness_list[0][1]

    com_dists = []
    com_energies = []

    for p_energy, p_coords in boundness_list[1:]:
        dist_to_com = most_bound_part - p_coords

        # todo get magnitude of dist_to_com
    
        com_dists.append(np.linalg.norm(dist_to_com.xyz.value))
        com_energies.append(p_energy)

    plt.plot(com_dists)
    plt.yscale('log')
    plt.show()

    plt.scatter(com_energies, com_dists)
    plt.ylabel('Distance to most bound particle (kpc)')
    plt.xlabel('Binding energy of the particle (erg)')
    plt.show()

def test_pot_mass(pot):

    accs = []
    for i in range(1, 100):
        r = i * u.kpc
        print('radius', i)
        #a = pot.acceleration(i * u.kpc)
        #accs.append((r, a))

        print(i, end = ' ')
        m = hernquist_menc(130.0075e10 * u.Msun, r,  32.089 * u.kpc)
        print('%1.3e' % m.value)
        
        #print(a)
        #print(pot.mass_enclosed(i * u.kpc))

        #print((-a * r * r / G).to(u.Msun))

    #print(accs)

    #for r, a in accs:
    #    print((-a * r * r / G).to(u.Msun))

def hernquist_menc(mtot, r, c):
    return mtot * r * r / (r + c) / (r + c)

def select_in_tidal_radius():
    sat_mass = 1e8
    
    pot = gp.HernquistPotential(m=130.0075e10*u.Msun,c = 32.089, units=galactic)

    particle_list = read_particles('parts_tidal_radius')

    particle_list = np.transpose(particle_list).tolist()

    #print(particle_list[0] * u.kpc)

    positions = [particle_list[i] for i in range(0, 3)] * u.kpc
    velocities = [particle_list[i] for i in range(3, 6)] * (u.km / u.s)

    w0 = gd.PhaseSpacePosition(pos=positions, vel=velocities)


    print('integrating ... ')
    start_time = time.time()
    orbit = gp.Hamiltonian(pot).integrate_orbit(w0,dt=-10.67, n_steps = 250)
    orbit_time = time.time() - start_time
    print('done integrating')
    print('Length of integration: %1.3f' % orbit_time)

    #orbit[0].plot().show()

    print(orbit.shape[1])

    com_p, com_v, com_index = calc_com(sat_mass / orbit.shape[1], orbit, pot)

    calc_dps(sat_mass / orbit.shape[1], orbit, pot, com_index)

def file_string(m, r, v, x, y, z, vx, vy, vz):
    return '%15.5e' % m + \
           '%15.5e' % r + \
           '%15.5e' % v + \
           '%15.5e' % x + \
           '%15.5e' % y + \
           '%15.5e' % z + \
           '%15.5e' % vx + \
           '%15.5e' % vy + \
           '%15.5e' % vz + '\n'

def write_tidal_radius_file(r_ratio = 0.5, sat_mass=1e8*u.Msun, in_file = 'centroid_part_1000', out_file_name = 'parts_tidal_radius'):
    print('Writing Tidal Radius File')
    print('args:')
    print('\tr_ratio', r_ratio)
    print('\tsat_mass', sat_mass)
    print('\tin_file', in_file)
    print('\tout_file_name', out_file_name)
    #sat_mass = 1e8 * u.Msun
    mw_mass = 130.0075e10*u.Msun
    c_param = 32.089
    sgr_r = 4.95309E+01
    sgr_v = 2.83513E+02
    
    pot = gp.HernquistPotential(m=mw_mass, c = c_param, units=galactic)

    particle_list = read_particles(file_name=in_file)

    particle_list = np.transpose(particle_list).tolist()

    #print(particle_list[0] * u.kpc)

    positions = [particle_list[i] for i in range(0, 3)] * u.kpc
    velocities = [particle_list[i] for i in range(3, 6)] * (u.km / u.s)

    w0 = gd.PhaseSpacePosition(pos=positions, vel=velocities)


    print('integrating ... ')
    start_time = time.time()
    orbit = gp.Hamiltonian(pot).integrate_orbit(w0,dt=-10.67, n_steps = 250)
    orbit_time = time.time() - start_time
    print('done integrating')
    print('Length of integration: %1.3f' % orbit_time)

    com_pos, com_vel, com_index = calc_com(sat_mass.value / 1000, orbit, pot)

    distances_to_com = [np.linalg.norm(n.xyz) for n in (orbit[0].pos - com_pos)]

    com_r = np.linalg.norm(com_pos.xyz)

    print('com_r', com_r)
    print('hernquist mass', hernquist_menc(mw_mass, com_r, c_param))
    

    tidal_radius = com_r * u.kpc * (sat_mass / 3 / hernquist_menc(mw_mass, com_r, c_param))**(1/3)

    print('Tidal radius:', tidal_radius)
    selected_indices = []
    for p_index in range(len(distances_to_com)):
        curr_dist = distances_to_com[p_index]
        if curr_dist < (tidal_radius * r_ratio).value:
            selected_indices.append(p_index)
            print(p_index, curr_dist)

    print(selected_indices)

    outfile = open(out_file_name, 'w')

    outlines = ['%8i' % len(selected_indices) + '\n']
    particle_mass = (sat_mass / len(selected_indices)).value

    for p_index in selected_indices:
        curr_pos = orbit[0].pos[p_index]
        curr_vel = orbit[0].vel[p_index]

        curr_file_line = file_string(particle_mass, sgr_r, sgr_v,\
                curr_pos.x.value, curr_pos.y.value, curr_pos.z.value,\
                curr_vel.d_x.value, curr_vel.d_y.value, curr_vel.d_z.value)
        outlines.append(curr_file_line)

    outfile.writelines(outlines)
    outfile.close()

    for line in outlines:
        print(line, end='')

    print('done!')

    
def __main__():
    test_c = 32.089
    test_m = 130.0075
    pot = gp.HernquistPotential(m=test_m*1e10*u.Msun,c =test_c, units=galactic)

    #plot_pot(pot)

    #integrate()
    #test_pot_mass(pot)

    #write_tidal_radius_file()
    #select_in_tidal_radius()
    #write_tidal_radius_file(sat_mass=1e9*u.Msun, in_file='heavy_sag_7_10', out_file_name='heavy_sag_core')
    integrate(input_file='heavy_sag_core',sat_mass=1e9)

__main__()
