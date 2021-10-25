import os, h5py
from PIL import Image
import numpy as  np

from mp3.submitted import todo_find_bestsize

def classify(Xtrain, Ytrain, Xdev, Ydev, Xtest):
    '''
    Ytest = classify(Xtrain, Ytrain, Xdev, Ydev, Xtest)

    Use any technique you like to train a classifier with the training set,
    and then return the correct class labels for the test set.
    Extra credit points are provided for beating various thresholds above 50%.

    Xtrain (NTRAIN x NDIM) - data matrix.  You may assume that NDIM > NTOKS
    Ytrain (NTRAIN) - list of class indices
    Xdev (NDEV x NDIM) - data matrix.
    Ydev (NDEV) - list of class indices
    Xtest (NTEST x NDIM) - data matrix.
    '''
    ctrain, cdev, ctest = todo_center_datasets(Xtrain, Xdev, Xtest, todo_dataset_mean(Xtrain))
    V, Lambda = todo_find_transform(ctrain)
    ttrain, tdev, ttest = todo_transform_datasets(ctrain, cdev, ctest, V)
    Dtraintrain = todo_distances(ttrain, ttrain, ttrain.shape[1])
    Dtraindev = todo_distances(ttrain, tdev, ttrain.shape[1])
    hyps = todo_nearest_neighbor(Ytrain, Dtraindev)
    bestsize, accuracies = todo_find_bestsize(ttrain, tdev, Ytrain, Ydev, Lambda)
    hyps = todo_nearest_neighbor(Ytrain, todo_distances(ttrain, ))

    
def todo_dataset_mean(X):
    '''
    mu = todo_dataset_mean(X)
    Compute the average of the rows in X (you may use any numpy function)
    X (NTOKSxNDIMS) = data matrix
    mu (NDIMS) = mean vector
    '''
    return np.mean(X, axis=0)

def todo_center_datasets(train, dev, test, mu):
    '''
    ctrain, cdev, ctest = todo_center_datasets(train, dev, test, mu)
    Subtract mu from each row of each matrix, return the resulting three matrices.
    '''
    return train - mu, dev - mu, test - mu

def todo_find_transform(X):
    '''
    V, Lambda = todo_find_transform(X)
    X (NTOKS x NDIM) - data matrix.  You may assume that NDIM > NTOKS
    V (NDIM x NTOKS) - The first NTOKS principal component vectors of X
    Lambda (NTOKS) - The  first NTOKS eigenvalues of the covariance or gram matrix of X

    Find and return the PCA transform for the given X matrix:
    a matrix in which each column is a principal component direction.
    You can assume that the # data is less than the # dimensions per vector,
    so you should probably use the gram-matrix method, not the covariance method.
    Standardization: Make sure that each of your returned vectors has unit norm,
    and that its first element is non-negative.
    Return: (V, Lambda)
      V[:,i] = the i'th principal component direction
      Lambda[i] = the variance explained by the i'th principal component

    V and Lambda should both be sorted so that Lambda is in descending order of absolute
    value.  Notice: np.linalg.eig doesn't always do this, and in fact, the order it generates
    is different on my laptop vs. the grader, leading to spurious errors.  Consider using 
    np.argsort and np.take_along_axis to solve this problem, or else use np.linalg.svd instead.
    '''
    u, s, v = np.linalg.svd(X, full_matrices=False)
    return v.T, s*s

def todo_transform_datasets(ctrain, cdev, ctest, V):
    '''
    ttrain, tdev, ttest = todo_transform_datasets(ctrain, cdev, ctest, V)
    ctrain, cdev, ctest are each (NTOKS x NDIMS) matrices (with different numbers of tokens)
    V is an (NDIM x K) matrix, containing the first K principal component vectors
    
    Transform each x using transform, return the resulting three datasets.
    '''
    return ctrain.dot(V), cdev.dot(V), ctest.dot(V)

def todo_distances(train,test,size):
    '''
    D = todo_distances(train, test, size)
    train (NTRAINxNDIM) - one training vector per row
    test (NTESTxNDIM) - one test vector per row
    size (scalar) - number of dimensions to be used in calculating distance
    D (NTRAIN x NTEST) - pairwise Euclidean distances between vectors

    Return a matrix D such that D[i,j]=distance(train[i,:size],test[j,:size])
    '''
    D = np.zeros(shape=(train.shape[0], test.shape[0]))
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            D[i, j] = np.linalg.norm(train[i, :size] - test[j, :size])
    return D

def todo_nearest_neighbor(Ytrain, D):
    '''
    hyps = todo_nearest_neighbor(Ytrain, D)
    Ytrain (NTRAIN) - a vector listing the class indices of each token in the training set
    D (NTRAIN x NTEST) - a matrix of distances from train to test vectors
    hyps (NTEST) - a vector containing a predicted class label for each test token

    Given the dataset train, and the (NTRAINxNTEST) matrix D, returns
    an int numpy array of length NTEST, specifying the person number (y) of the training token
    that is closest to each of the NTEST test tokens.
    '''
    hyps = np.array([Ytrain[np.argmin(D[:, i])] for i in range(D.shape[1])])
    return hyps

def to_greyscale(X):
    X_ = np.zeros(shape=(X.shape[0], X.shape[1]//3))
    for i, image in enumerate(X):
        image = image.reshape(250, 250, 3).mean(axis=2)
        X_[i] = image.reshape(np.prod(image.shape))
    return X_[i]