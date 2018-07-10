import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import statistics

verbose = False
kms = 1e5

def mass_enc(r, a):
    return r**3/(r**2+a**2)**(3.0/2)
  
def mass_prime(r, a):
    return 3 * r**2 / (r**2+a**2)**(5.0/2)
  
def newton_next(r_n, u, a):
    r_n = float(r_n)
    u = float(u)
    a = float(a)
    return r_n - (mass_enc(r_n, a) - u) / mass_prime(r_n, a)
  
def run_newton(n, u, a):
    guess = 1.0
    for i in range(n):
        guess = newton_next(guess, u, a)
        if verbose:
            print(guess)
    return guess

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

def generate_random():

    show_plots = False

    num_particles = 100
    scale_radius = 1.55

    sgr_mass = 2.0e+10
    sgr_r = 4.95309E+01
    sgr_v = 2.83513E+02
    # sgr_coords = (-7.8e-1, -4.15500e-1, -2.695e1)
    # sgr_velocity = (-2.92994e+00, -2.00734E+02, 2.00192E+02)
    sgr_coords = (1.9e1, 2.7e0, -6.9e0)
    sgr_velocity = (2.3e2, -3.5e1, 1.95e2)
    
    values = [random.uniform(0, 1) for i in range(num_particles)]

    thetas = [random.uniform(0, 1) * 2.0 * math.pi for i in range (num_particles)]
    phis = [math.acos(1 - 2 * random.uniform(0,1)) for i in range (num_particles)]

    radii = [run_newton(10, v, scale_radius / 1.3) for v in values]

    xs = [sgr_coords[0] + radii[i] * math.sin(phis[i]) * math.cos(thetas[i]) for i in range(num_particles)]
    ys = [sgr_coords[1] + radii[i] * math.sin(phis[i]) * math.sin(thetas[i]) for i in range(num_particles)]
    zs = [sgr_coords[2] + radii[i] * math.cos(phis[i]) for i in range(num_particles)]

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    if show_plots:
        plt.scatter(xs, ys)
        plt.show()

        plt.scatter(xs, zs)
        plt.show()

        plt.scatter(ys, zs)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(xs, ys, zs)

        plt.show()

    outlines = ['%8i' % num_particles + '\n']
    for i in range(num_particles):
        curr_file_line = file_string(sgr_mass / num_particles, sgr_r, sgr_v,\
                                xs[i], ys[i], zs[i],\
                              sgr_velocity[0], sgr_velocity[1], sgr_velocity[2])
        outlines.append(curr_file_line)
    outf = open('test_100', 'w')

    outf.writelines(outlines)
    outf.close()


def random_select_parts():

    sgr_mass = 2.0e+10
    sgr_r = 4.95309E+01
    sgr_v = 2.83513E+02
    
    num_particles = 100
    print("Selecting", num_particles, "from the Sag dwarf distribution")

    parts_file = open('sag_particles_265_ago.txt')
    # parts_file = open('sag_particles_now.txt')

    particle_list = []
    next(parts_file)
    for line in parts_file:
        curr_data = [float(val) for val in line.split()[1:]]
        particle_list.append(curr_data)

    random.shuffle(particle_list)
    particle_list = particle_list[:num_particles]

    outlines = ['%8i' % num_particles + '\n']
    for i in range(num_particles):
        curr_file_line = file_string(sgr_mass / num_particles, sgr_r, sgr_v,\
                                    particle_list[i][0],\
                                    particle_list[i][1],\
                                    particle_list[i][2],\
                                    particle_list[i][3],\
                                    particle_list[i][4],\
                                    particle_list[i][5])
        outlines.append(curr_file_line)
    outf = open('random_sag_100_265_right_units', 'w')

    outf.writelines(outlines)
    outf.close()

def random_select_parts_correct():

    sgr_mass = 2.0e+8
    sgr_r = 4.95309E+01
    sgr_v = 2.83513E+02
    
    num_particles = 100
    print("Selecting", num_particles, "from the Sag dwarf distribution")

    parts_file = open('ben_sag_05_30\\snapshot_108.hdf5.ascii')
    # parts_file = open('sag_particles_now.txt')

    particle_list = []
    next(parts_file)
    for line in parts_file:
        if line[0] == '#':
            continue
        curr_data = [float(val) for val in line.split()[0:]]
        particle_list.append(curr_data)

    random.shuffle(particle_list)
    particle_list = particle_list[:num_particles]

    outlines = ['%8i' % num_particles + '\n']
    for i in range(num_particles):
        curr_file_line = file_string(sgr_mass / num_particles, sgr_r, sgr_v,\
                                    particle_list[i][0],\
                                    particle_list[i][1],\
                                    particle_list[i][2],\
                                    particle_list[i][3],\
                                    particle_list[i][4],\
                                    particle_list[i][5])
        outlines.append(curr_file_line)
    outf = open('sag_100_05_30', 'w')

    outf.writelines(outlines)
    outf.close()    

def calc_com():
    
    sgr_mass = 2.0e+8
    sgr_r = 4.95309E+01
    sgr_v = 2.83513E+02

    print("Calculating the center of mass of these particles")

    parts_file = open('ben_sag_05_30\\snapshot_108.hdf5.ascii')
    # parts_file = open('sag_particles_now.txt')

    particle_list = []
    next(parts_file)
    for line in parts_file:
        if line[0] == '#':
            continue
        curr_data = [float(val) for val in line.split()[0:]]
        particle_list.append(curr_data)

    outlines = ['%8i' % 1 + '\n']

    means = []
    means.append(statistics.mean([part[0] for part in particle_list]))
    means.append(statistics.mean([part[1] for part in particle_list]))
    means.append(statistics.mean([part[2] for part in particle_list]))
    means.append(statistics.mean([part[3] for part in particle_list]))
    means.append(statistics.mean([part[4] for part in particle_list]))
    means.append(statistics.mean([part[5] for part in particle_list]))
    

    outlines.append(file_string(\
        sgr_mass, sgr_r, sgr_v,\
        means[0], means[1], means[2],\
        means[3], means[4], means[5]\
        ))

    outf = open('com_sag_100_05_30', 'w')

    outf.writelines(outlines)
    outf.close()
    

def __main__():
    #random_select_parts_correct()
    calc_com()
  
__main__()
