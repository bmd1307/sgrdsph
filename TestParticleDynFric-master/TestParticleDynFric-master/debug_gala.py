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


def plot_pot(potential):
    pot_file = open('potn.txt')

    counter = 1
    lines = []
    for line in pot_file:
        if line[0] == '#':
            continue
        rad, ben_phi = [float(val) for val in line.split()]
        pot_phi = potential.acceleration( \
            [rad, 0, 0] * u.kpc).to(u.cm / u.s / u.s).value[0][0]
        lines.append("%15.5e %15.5e %15.5e %15.5e" % \
                     (rad, ben_phi, pot_phi, ben_phi / pot_phi))
        if counter % 100 == 0:
            print('.', end='')
        counter = counter + 1

    pot_file.close()

    for i in range(0, 50):
        print(lines[i * 100])
