# Vectorization of persistence barcode with applications in pattern classification of porous structures (Computers & Graphics 2020)
This is an implement of the algorithm in the paper titled
"Vectorization of persistence barcode with applications in pattern classification of porous structures"

# requirements

## Operating System
Window (The authors used Windows 7 64-bit)

## Matlab
1. Matlab version: MATLAB R2017b or later.
2. Include 'drtoolbox' provided in the directory.

## Python 
Python version: Python 3.6
Visit the website https://www.python.org/downloads/release/python-362/ to install Python 3.6.2 by downloading "Windows x86-64 executable installer" and installing it.

## Python Packages
Required Python packages include: ripser, cython, numpy, matplotlib, scikit-tda, tqdm, seaborn, pandas, scikit-learn

Run the script ./install.sh to automatically install the Python packages.


# Code and files
## Computing Barcodes
python files:

BC_compute.py: To compute barcodes of point clouds.


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
1. main_porous.m 
2. main_dynam2D 
3. main_timeseries 


# Instructions

## Set path
Set the folder to the folder where the code is located.

## data1
This data set is about porous classification. The results are shown in Table 2 and Section 6.

Data:

The original training and test data, i.e., 3D point clouds, are given in the format of txt file in the directory "Sample_training" and "Sample_test".

The barcodes of training and test data are given in the directory of "BC_training" and "BC_test".

Run "main_porous.m" in MATLAB, and then input "Y" or "y" to use the dimensionality reduction, or "N" or "n" not to use it.

The key parameters are printed on the screen, such as bound value, and reduced dimension.

Output: 

The classification accuracy on four classifiers with the corresponding hyper-parameter, namely, Logistic Regression, KNN, Kernel SVM (SVC), and Linear SVM (LinearSVC).


## data3
This data set is about the determination of the parameters of the 2D dynamical system proposed by Adams et al. (2017). 

Data:

The original training and test data, i.e., 2D point clouds, are given in the format of txt file in the directory "training_data" and "test_data".

The barcodes of training and test data are given in the directory of "BC_training" and "BC_test".

Run "main_dynam2D.m" in MATLAB, and then input "Y" or "y" to use the dimensionality reduction, or "N" or "n" not to use it.

The key parameters are printed on the screen, such as bound value, and reduced dimension.

Output: 

The classification accuracy on four classifiers with the corresponding hyper-parameter, namely, Logistic Regression, KNN, Kernel SVM (SVC), and Linear SVM (LinearSVC).


## data4
This data set is about time-series for multi-source signals provided in UCR archive. Four of the data sets in our experiments are offered, including BirdChicken, Earthquakes, ECG200, and FreezerRegularTrain. 

There are four sub-directories, namely, BirdChicken, Earthquakes, ECG200, and FreezerRegularTrain.

In each directory, the original training and test data, i.e., high-dimensional point clouds, are given in the format of txt file in the directory "TRAIN" and "TEST".

The barcodes of training and test data are given in the directory of "BC_train" and "BC_test", respectively. 

1. Run "main_timeseries.m"
2. Input the name of data set, for exaple "BirdChicken".
3. Input "Y" or "y" to use the dimensionality reduction, or "N" or "n" not to use it.

Output: 

The classification accuracy on four classifiers with the corresponding hyper-parameter, namely, Logistic Regression, KNN, Kernel SVM (SVC), and Linear SVM (LinearSVC).


# Notes
1. For fast computing, we set the discretization parameter "sam_num" to be 8, which is 10 in our experiments. The accuracy does not lose much. 
2. Note that the results of data3 and data4 are shown in Figure 2(a) in the paper.
3. Note that we use Matlab to call the command line to execute the python file for computing the barcodes of point clouds.

If, unfortunately, some errors occur in this process, execute the commands given by the notes in Matlab .m file in the command console. 

For example, when running "main_porous.m", if it throws an error when computing barcodes stored in "BC_training, pause and parse the command "python BC_compute.py ./data1/Sample_training ./data1/BC_training" on the command console. Then rerun the "main_porous.m".

For convenience, we provide both original data and barcodes stored in "BC_train" and "BC_test" in each data set.