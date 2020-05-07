import os
from ripser import Rips
import numpy as np
import sys

sourcepath = str(sys.argv[1])
targetpath = str(sys.argv[2])
files = sorted(os.listdir(sourcepath))
rips = Rips()

for file in files:
    filepath = os.path.join(sourcepath,file)
    data = np.loadtxt(filepath)
    dgm = rips.fit_transform(data)
    np.savetxt(targetpath+'/'+file,np.array(dgm[1]),fmt='%s')