## Vectorization of persistence barcode with applications in pattern classification of porous structures (Computers & Graphics 2020)
This is an implement of generating feature vectors of persistence rank function (PRF) 
for vectorizing barcodes.

# requirements

## Matlab
1. Install Parallel Computing Toolbox.
2. Include 'drtoolbox' provided in the directory.

## Python 3
Install the package 'ripser':
pip install cython
pip install ripser

Install some packages for classification tasks:
pip install numpy matplotlib scikit-tda tqdm seaborn pandas scikit-learn


For more information on 'ripser', please refer to https://pypi.org/project/ripser/

# Code and files
## Computing Barcodes
python files:
BC_compute.py: To compute barcodes of point clouds.
computeImgBC.py: To compute barcodes w.r.t. low-star filtration of gray images.

## Computing Feature Vectors for Vectorizing Barcodes
haarBasisM.m
PersBettiSur.m 
haarDecomposition2DFunc.m

## Dimensionality Reduction
Directory:
drtoolbox

Files:
pca.m
A_Isomap.m
connect_components.m
find_NN_supervised.m
out_of_sample_new.m
Supervised_Isomap.m

## Classification
python file:
classifiers.py

## main function
1. main_porous.m (See data1)
2. main_GenImg.m (See data2)
3. main_dynam2D.m (See data3)
4. main_timeseries (See data4)

# Note
For fast computing, we set the discretization parameter N (Algorithm 3) to be 8, which is 10 in our experiments. The accuracy does not lose much. 


# Data sets
Due to storage limitation, the barcodes of data are provided for computing feature vectors.
The original data, such as the obj files of porous models and point clouds, will be public after paper reviewing.
## data1
This data set is about porous classification. The barcodes of both training and test data are provided.
To run the Matlab function 'main_porous' to obtain the feature vectors and classification accuracy on four classifiers.

## data2
This data set shows the random images of the Brownian motion. The generated images are too large to upload, so the barcodes and feature vectors are provided for computing classification accuracy.
To run the Matlab function 'main_GenImg' to obtain the classification accuracy.

## data3
This data set is about the determination of the parameters of the 2D dynamical system proposed by Adams et al. (2017). The whole data are provided in the directory 'data3'.
To run the Matlab function 'main_dynam2D' to obtain the classification accuracy.

## data4
This data set is about time-series for multi-source signals provided in UCR archive. Four of the data sets in our experiments are offered, including Beef, CBF, ChlorineConcentration, and FreezerRegularTrain. 
To run the Matlab function 'main_timeseries' and input the directory name, such as 'Beef', to obtain the classification accuracy.