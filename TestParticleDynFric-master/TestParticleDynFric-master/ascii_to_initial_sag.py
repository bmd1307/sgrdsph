
def file_string(x, y, z, vx, vy, vz, r =  4.95309E+01, v = 2.83513E+02, m = 1e6):
    return '%15.5e' % m + \
           '%15.5e' % r + \
           '%15.5e' % v + \
           '%15.5e' % x + \
           '%15.5e' % y + \
           '%15.5e' % z + \
           '%15.5e' % vx + \
           '%15.5e' % vy + \
           '%15.5e' % vz + '\n'

def read_file(file_name, out_name):
    ascii_file = open(file_name)

    particle_list = []
    for line in ascii_file:
        if line[0] == '#':
            continue

        particle_list.append([float(val) for val in line.split()])



    outlines = ['%8i' % len(particle_list) + '\n']
    for i in range(len(particle_list)):
        curr_file_line = file_string(particle_list[i][0], \
                                     particle_list[i][1], \
                                     particle_list[i][2], \
                                     particle_list[i][3], \
                                     particle_list[i][4], \
                                     particle_list[i][5], \
                                     m = particle_list[i][6])
        outlines.append(curr_file_line)
    outf = open(out_name, 'w')

    outf.writelines(outlines)
    outf.close()

    for line in outlines[0:10]:
        print(line, end='')

    print('...S')

def __main__():
    read_file('snapshot_106.hdf5.ascii', 'heavy_sag_7_10')

__main__()