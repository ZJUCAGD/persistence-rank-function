import numpy as np
import sys
import os

from ripser import ripser, lower_star_img

sourcePath = str(sys.argv[1])
targetPath = str(sys.argv[2])

files = sorted(os.listdir(sourcePath))

for file in files:
    filePath = os.path.join(sourcePath,file)
    data = np.loadtxt(filePath)
    dgm = lower_star_img(data)
    np.savetxt(targetPath+'/'+file,np.array(dgm[:-1]),fmt='%s')
