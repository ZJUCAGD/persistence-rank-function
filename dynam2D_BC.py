import scipy.io as scio # to load matlab file .mat
import numpy as np
import numpy as np
from ripser import Rips
import os

orbits1 = scio.loadmat('./data3/dynamic2d/orbits1')['orbits1']
orbits2 = scio.loadmat('./data3/dynamic2d/orbits2')['orbits2']
orbits3 = scio.loadmat('./data3/dynamic2d/orbits3')['orbits3']
orbits4 = scio.loadmat('./data3/dynamic2d/orbits4')['orbits4']
orbits5 = scio.loadmat('./data3/dynamic2d/orbits5')['orbits5']
orbits = [orbits1, orbits2, orbits3, orbits4, orbits5]

os.mkdir('./data3/training_data')
os.mkdir('./data3/test_data')
tr_labels = []
ts_labels= []
for i, orbit in enumerate(orbits):
    for j in range(40):
        tr_data = orbit[1000*j:1000*(j+1), :]
        tr_labels.append(i)
        tr_data = np.array(tr_data)
        np.savetxt('./data3/training_data/{}_{}.txt'.format(i,j), tr_data, fmt='%s')
    for j in range(40,50):
        ts_data = orbit[1000*j:1000*(j+1), :]
        ts_labels.append(i)
        ts_data = np.array(ts_data)
        np.savetxt('./data3/test_data/{}_{}.txt'.format(i,j), ts_data, fmt='%s')
        
tr_labels = np.array(tr_labels)
ts_labels = np.array(ts_labels)
np.savetxt('./data3/training_labels.txt', tr_labels, fmt='%s')
np.savetxt('./data3/test_labels.txt', ts_labels, fmt='%s')