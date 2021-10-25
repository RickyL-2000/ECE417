import os, h5py
from PIL import Image
import numpy as  np

###############################################################################
# TODO: here are the functions that you need to write
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

def todo_compute_accuracy(Ytest, hyps):
    '''
    ACCURACY, CONFUSION = todo_compute_accuracy(TEST, HYPS)
    TEST (NTEST) - true label indices of each test token
    HYPS (NTEST) - hypothesis label indices of each test token
    ACCURACY (scalar) - the total fraction of hyps that are correct.
    CONFUSION (4x4) - confusion[ref,hyp] is the number of class "ref" tokens (mis)labeled as "hyp"
    '''
    confusion = np.zeros(shape=[4, 4])
    for i in range(4):
        for j in range(4):
            confusion[i, j] = np.sum(hyps[Ytest == i] == j)
    accuracy = np.sum(np.diagonal(confusion)) / np.sum(confusion)
    return accuracy, confusion

def todo_find_bestsize(ttrain, tdev, Ytrain, Ydev, variances):
    '''
    BESTSIZE, ACCURACIES = todo_find_bestsize(TTRAIN, TDEV, YTRAIN, YDEV, VARIANCES)
    TTRAIN (NTRAINxNDIMS) - training data, one vector per row, PCA-transformed
    TDEV (NDEVxNDIMS)  - devtest data, one vector per row, PCA-transformed
    YTRAIN (NTRAIN) - true labels of each training vector
    YDEV (NDEV) - true labels of each devtest token
    VARIANCES - nonzero eigenvectors of the covariance matrix = eigenvectors of the gram matrix

    BESTSIZE (scalar) - the best size to use for the nearest-neighbor classifier
    ACCURACIES (NTRAIN) - accuracy of dev classification, as function of the size of the NN classifier

    The only sizes you need to test (the only nonzero entries in the ACCURACIES
    vector) are the ones where the PCA features explain between 92.5% and
    97.5% of the variance of the training set, as specified by the provided
    per-feature variances.  All others should be zero.
    '''
    accuracies = np.zeros(shape=ttrain.shape[0])
    for K in range(accuracies.shape[0]):
        if 92.5 < 100 * np.sum(variances[:K+1]) / np.sum(variances) < 97.5:
            hyps = todo_nearest_neighbor(Ytrain, todo_distances(ttrain, tdev, K+1))
            accuracy, _ = todo_compute_accuracy(Ydev, hyps)
            accuracies[K] = accuracy
    bestsize = np.argmax(accuracies) + 1
    return bestsize, accuracies

