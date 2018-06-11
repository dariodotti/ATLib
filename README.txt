### ATLIB ####

## I Version ##

INPUT: path of feature matrix
OUTPUT: classification accuracy of normal vs abnormal trajectories


It is possible to change the parameters (e.g. number of cluster, normal/abnormal labels) 
only in the ATLib_main.py.

ATLib_classifiers.py contains all the machine learning methods in order to perform clustering, training, predicting data.
 

## II Version ##

This library was extended with methods for generating patch features from given trajectory points, as well as methods to call deep neural network
library such as Theanets.
Prerequisite: Install theanets and Theano

To generate patch feature from trajectory call: traj_to_patch(x_f,y_f,size_mask)
To train and use the autoencoder methods, see the Autoencoder.py class