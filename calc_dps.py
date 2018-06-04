#!/usr/bin/python

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

# This is a class to hold a phase-space vector
class PSVector:
	def __init__(self, x, y, z, vx, vy, vz):
		self.x = x
		self.y = y
		self.z = z
		self.vx = vx
		self.vy = vy
		self.vz = vz

	# return the magnitude of the phase-space vector ( hypot(posiiton, velocity) )
	def mag(self):
		return (self.x * self.x + \
			self.y * self.y + \
			self.z * self.z + \
			self.vx * self.vx + \
			self.vy * self.vy + \
			self.vz * self.vz) ** 0.5

	# returns the distance between the positions of two phase-space vectors
	def distance(self, vect2):
		return ((self.x - vect2.x)**2 + (self.y - vect2.y)**2 + (self.z - vect2.z)**2)**0.5

def __main__():
	out_graph_name = None

	if len(sys.argv) != 3 and len(sys.argv) != 4:
		print "Usage: ./calc_dps <com file> <particle file> [<out plot file>]"
		quit()
	
	if len(sys.argv) == 4:
		out_graph_name = sys.argv[3]
	com_file = open(sys.argv[1])
	part_file = open(sys.argv[2])

	timesteps = []
	particle_indices = []

	#dictionary from the timestep to (PSVector, Rtide, vesc)
	com_dict = {}

	#dictionary from particle index to (dictionary from timestep to PSVector)
	part_dict = {}

	#a dictionary from particle indices to ps vectors
	ps_vect_dict = {}

	#min dps matrix
	min_dps_mat = []

	# 1. read in the particle data from the files:

	for line in com_file:
		points = [float(v) for v in line.split()]
		
		curr_ps_vec = PSVector(points[2], points[3], points[4], points[5], points[6], points[7])
		curr_timestep = points[1]

		timesteps.append(curr_timestep)

		com_dict[curr_timestep] = (curr_ps_vec, points[8], points[9])

	for line in part_file:
		points = [float(v) for v in line.split()]
		
		curr_ps_vec = PSVector(points[2], points[3], points[4], points[5], points[6], points[7])
		curr_timestep = points[1]
		curr_particle_index = int(points[0])

		if not (curr_particle_index in part_dict):
			part_dict[curr_particle_index] = {}
			particle_indices.append(curr_particle_index)

		part_dict[curr_particle_index][curr_timestep] = curr_ps_vec

	# 2. calculate the normalized ps-vector for each particle at each timestep

	# for each particle
	for p_index in particle_indices:
		ps_vect_dict[p_index] = {}
		
		# for each timestep
		for ts in timesteps:
			# get the ps-vector, tidal radius and the escape velocity for the COM
			curr_com_vector, curr_rtide, curr_vesc = com_dict[ts]
			# get the ps-vector for the particle
			curr_part_vector = part_dict[p_index][ts]

			# calculate the components of the normalized ps-vector
			curr_x = (curr_part_vector.x - curr_com_vector.x) / curr_rtide
			curr_y = (curr_part_vector.y - curr_com_vector.y) / curr_rtide
			curr_z = (curr_part_vector.z - curr_com_vector.z) / curr_rtide
			curr_vx = (curr_part_vector.vx - curr_com_vector.vx) / curr_vesc
			curr_vy = (curr_part_vector.vy - curr_com_vector.vy) / curr_vesc
			curr_vz = (curr_part_vector.vz - curr_com_vector.vz) / curr_vesc

			# store this ps-vector
			ps_vect_dict[p_index][ts] = PSVector(curr_x, curr_y, curr_z, curr_vx, curr_vy, curr_vz)

	# 3. find the minimum ps-vector magnitude for each particle
	#	append that vector to the dps matrix	

	for p_index in particle_indices:
		min_ts = -1
		min_mag = 1e99
		for ts in timesteps:
			curr_mag = ps_vect_dict[p_index][ts].mag()

			if curr_mag < min_mag:
				min_mag = curr_mag
				min_ts = ts
		min_vect = ps_vect_dict[p_index][min_ts]
		min_dps_mat.append([min_vect.x, min_vect.y, min_vect.z, min_vect.vx, min_vect.vy, min_vect.vz])

	# 4. find the covariance matrix (Sigma_n in Price-Whelan, Johnston 2013)

	min_dps_mat = np.array(min_dps_mat).transpose()
	
	cov_mat = np.cov(min_dps_mat)

	print "cov mat shape", cov_mat.shape
	print cov_mat

	# 5. calculate the determinant of this covariant matrix

	det_mat = np.linalg.det(cov_mat)
	print "determinant of the matrix", det_mat

	# 6. calculate the log of the determinant of the matrix

	if det_mat > 0:
		ln_det = math.log(det_mat)
		print "log det:", ln_det
	else:
		print "cannot compute log of negative number"

	# find the minimum distances here
	min_dist = 1e99
	
	if False:
		for ts in timesteps:
			for i in range(0, len(particle_indices)):
				for j in range(i + 1, len(particle_indices)):
					p1 = part_dict[particle_indices[i]][ts]
					p2 = part_dict[particle_indices[j]][ts]

					curr_distance = p1.distance(p2)
					if curr_distance < min_dist:
						min_dist = curr_distance
			print(min_dist)

	zero_vector = PSVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	
	if False:
		max_dist = 0.0
		for ts in timesteps:
			for i in range(0, len(particle_indices)):
				p1 = part_dict[particle_indices[i]][ts]
	
				curr_distance = p1.distance(zero_vector)
				if curr_distance > max_dist:
					max_dist = curr_distance
			print(max_dist)

	# if a file name for an image was given to the program, save the Dps plot
	
	if out_graph_name is not None:
		print "Saving to", out_graph_name
		for pi in particle_indices[0:10]:
			plt.plot(timesteps, [ps_vect_dict[pi][ts].mag() for ts in timesteps], c='gray', lw=0.5)
		plt.yscale('log')
		plt.ylim([0.1, 500])
		plt.xlim([0, 2650])
		plt.savefig(out_graph_name, bbox_inches='tight')
	
__main__()
