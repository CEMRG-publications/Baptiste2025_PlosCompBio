import os
import numpy as np
from GPErks.serialization.labels import read_labels_from_file


def main():

    basefolder = '/home/tmb119/Dropbox/code/GPErks_library/ctcrt24_ani/'
    waveno = 1
    loadpath = basefolder + 'hm_output/wave{}/'.format(waveno)
    
    vars = np.loadtxt(loadpath + "PVn_" + str(waveno) + ".txt", dtype = float)

    print(np.shape(vars))

    print(np.mean(vars, axis = 0))
    print(np.median(vars, axis = 0))
    print(np.median(vars, axis = 1))

if __name__ == '__main__':
	main()
