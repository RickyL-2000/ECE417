import os, h5py
import numpy as  np
from scipy.stats import multivariate_normal

def recognize(Xtrain, Xdev, Xtest):
    '''Use any technique you like to train a recognizer with the training set,
    and then return the correct class labels for the test set.
    Extra credit points are provided for beating various thresholds above 33% accuracy.

    Inputs:
    Xtrain (dict of lists of (nframes,nceps) arrays):
        Xtrain[y][n][t,:] = t'th frame of n'th training utterance of word Y=y
    Xdev (dict of lists of (nframes,nceps) arrays):
        Xdev[y][n][t,:] = t'th frame of n'th devtest utterance of word Y=y
    Xtest (list of (nframes,nceps)  arrays):
        Xtest[n][t,:] = t'th frame of the n'th test utterance

    Returns:
    Y_hat (list of scalar strings): 
        Predicted label of each of the test utterances

    Implementation Warning:
    For the hidden dataset, the word labels will not be ['1','2','3'].
    Instead of hardcoding ['1','2','3'], use Xtrain.keys() or Xdev.keys().
    '''
    labels = list(Xtrain.keys())
    nstates = 5
    Lambda = {}
    B_dict = {y:[] for y in labels}
    print("initializing...")
    for y in labels:
        Lambda[y] = initialize_hmm(Xtrain[y], nstates)
    
    # sampling data set
    for y in labels:
        Xtrain[y] = Xtrain[y][: int(len(Xtrain[y])/2)]

    print("begin training...")
    epochs = 5
    for epoch in range(epochs):
        # stochastic
        sub_XTrain = {}
        for y in labels:
            N = len(Xtrain[y])
            sub_XTrain[y] = Xtrain[y][int(epoch * N / epochs): min(int((epoch+1) * N / epochs), N)]

        print(f"epoch {epoch}...")
        B_dict = { y:[] for y in labels }
        for y in labels:
            for X in sub_XTrain[y]:
                B = observation_pdf(X, Lambda[y][1], Lambda[y][2])
                B_dict[y].append(B)
        Alpha_dict = { y:[] for y in labels }
        G_dict = { y:[] for y in labels }
        for y in labels:
            for B in B_dict[y]:
                Alpha_Hat, G = scaled_forward(Lambda[y][0], B)
                Alpha_dict[y].append(Alpha_Hat)
                G_dict[y].append(G)
        Beta_dict = { y:[] for y in labels }
        for y in labels:
            for B in B_dict[y]:
                Beta_Hat = scaled_backward(Lambda[y][0], B)
                Beta_dict[y].append(Beta_Hat)
        Gamma_dict = { y:[] for y in labels }
        Xi_dict = { y:[] for y in labels }
        for y in labels:
            for n, (B, Alpha_Hat, Beta_Hat) in enumerate(zip(B_dict[y], Alpha_dict[y], Beta_dict[y])):
                Gamma, Xi = posteriors(Lambda[y][0], B, Alpha_Hat, Beta_Hat)
                Gamma_dict[y].append(Gamma)
                Xi_dict[y].append(Xi)
        
        # E
        expectations = {}
        for y in labels:
            expectations[y] = E_step(np.concatenate(sub_XTrain[y]), np.concatenate(Gamma_dict[y]), np.concatenate(Xi_dict[y]))
        
        # M
        regularizer = 1
        for y in labels:
            Lambda[y] = M_step(*expectations[y], regularizer)
    
    print("begin inferencing...")
    # inference
    for y in ['1','2','3']:
        logprob, Yhat = myrecognize(Xtest, Lambda)
    
    return Yhat

    raise NotImplementedError("You need to write this!")

def initialize_hmm(X_list, nstates):
    '''Initialize hidden Markov models by uniformly segmenting input waveforms.

    Inputs:
    X_list (list of (nframes[n],nceps) arrays): 
        X_list[n][t,:] = feature vector, t'th frame of n'th waveform, for 0 <= t < nframes[n]
    nstates (scalar): 
        the number of states to initialize

    Returns:
    A (nstates,nstates):
        A[i,j] = p(state[t]=j | state[t-1]=i), estimates as
        (# times q[t]=j and q[t-1]=i)/(# times q[t-1]=i).
    Mu (nstates,nceps):
        Mu[i,:] = mean vector of the i'th state, estimated as
        average of the frames for which q[t]=i.
    Sigma (nstates,nceps,nceps):
        Sigma[i,:,:] = covariance matrix, i'th state, est imated as
        unbiased sample covariance of the frames for which q[t]=i.
    
    Function:
    Initialize the initial HMM by dividing each feature matrix uniformly into portions for each state:
    state i gets X_list[n][int(i*nframes[n]/nstates):int((i+1)*nframes[n]/nstates,:] for all n.
    Then, from this initial uniform alignment, estimate A, MU, and SIGMA.

    Implementation Cautions:
    - For the last state, (# times q[t-1]=i) is not the same as (# times q[t]=i).
    - "Unbiased" means that you divide by N-1, not N.  In np.cov, that means "bias=False".
    '''
    A = np.zeros(shape=(nstates, nstates))
    A_denom = np.zeros(nstates)
    states = []
    nceps = X_list[0].shape[1]
    Mu = np.zeros(shape=(nstates, nceps))
    Sigma = np.zeros(shape=(nstates, nceps, nceps))
    A[-1, -1] = 1.0
    A_denom[-1] = 1.0
    for n in range(len(X_list)):
        nframes = X_list[n].shape[0]
        for i in range(nstates):
            if i < nstates - 1:
                A_denom[i] += min(int((i+1)*nframes/nstates), nframes) - int(i * nframes/nstates)
                A[i, i] += min(int((i+1)*nframes/nstates), nframes) - int(i * nframes/nstates) - 1
                A[i, i+1] += 1

            if n == 0:
                states.append(X_list[n][int(i*nframes/nstates): min(int((i+1)*nframes/nstates), nframes), :])
            else:
                states[i] = np.append(states[i], 
                                      X_list[n][int(i*nframes/nstates): min(int((i+1)*nframes/nstates), nframes), :], 
                                      axis=0)
    A_denom[A_denom == 0.0] = 1.0
    A /= A_denom.reshape(-1, 1)
    for i in range(nstates):
        Mu[i, :] = np.mean(states[i], axis=0)
        Sigma[i, :, :] = np.cov(states[i].T, bias=False)
    return A, Mu, Sigma

    raise NotImplementedError('You need to write this part!')

def observation_pdf(X, Mu, Sigma):
    '''Calculate the log observation PDFs for every frame, for every state.

    Inputs:
    X (nframes,nceps):
        X[t,:] = feature vector, t'th frame of n'th waveform, for 0 <= t < nframes[n]
    Mu (nstates,nceps):
        Mu[i,:] = mean vector of the i'th state
    Sigma (nstates,nceps,nceps):
        Sigma[i,:,:] = covariance matrix, i'th state

    Returns:
    B (nframes,nstates):
        B[t,i] = max(p(X[t,:] | Mu[i,:], Sigma[i,:,:]), 1e-100)

    Function:
    The observation pdf, here, should be a multivariate Gaussian.
    You can use scipy.stats.multivariate_normal.pdf.
    '''
    nframes, nceps = X.shape
    nstates = Sigma.shape[0]
    B = np.zeros(shape=(nframes, nstates))
    for n in range(nframes):
        for i in range(nstates):
            B[n, i] = max(multivariate_normal.pdf(X[n, :], Mu[i, :], Sigma[i, :, :]), 1e-100)
    return B

    raise NotImplementedError('You need to write this part!')
    
def scaled_forward(A, B):
    '''Perform the scaled forward algorithm.

    Inputs:
    A (nstates,nstates):
        A[i,j] = p(state[t]=j | state[t-1]=i)
    B (nframes,nstates):
        B[t,i] = p(X[t,:] | Mu[i,:], Sigma[i,:,:])

    Returns:
    Alpha_Hat (nframes,nstates):
        Alpha_Hat[t,i] = p(q[t]=i | X[:t,:], A, Mu, Sigma)
        (that's its definition.  That is not the way you should compute it).
    G (nframes):
        G[t] = p(X[t,:] | X[:t,:], A, Mu, Sigma)
        (that's its definition.  That is not the way you should compute it).

    Function:
    Assume that the HMM starts in state q_1=0, with probability 1.
    With that assumption, implement the scaled forward algorithm.
    '''
    nframes, nstates = B.shape
    Alpha_Hat = np.zeros(shape=(nframes, nstates))
    G = np.zeros(shape=nframes)
    # initialize
    Alpha_Hat[0, 0] = 1.0 * B[0, 0]
    G[0] = np.sum(Alpha_Hat[0, :])
    Alpha_Hat[0, :] /= G[0]
    # iterate
    for t in range(1, nframes):
        for j in range(nstates):
            for i in range(nstates):
                Alpha_Hat[t, j] += Alpha_Hat[t-1, i] * A[i, j] * B[t, j]
        G[t] = np.sum(Alpha_Hat[t, :])
        Alpha_Hat[t, :] /= G[t]
    return Alpha_Hat, G

    raise NotImplementedError('You need to write this part!')
    
def scaled_backward(A, B):
    '''Perform the scaled backward algorithm.

    Inputs:
    A (nstates,nstates):
        A[y][i,j] = p(state[t]=j | state[t-1]=i)
    B (nframes,nstates):
        B[t,i] = p(X[t,:] | Mu[i,:], Sigma[i,:,:])

    Returns:
    Beta_Hat (nframes,nstates):
        Beta_Hat[t,i] = p(X[t+1:,:]| q[t]=i, A, Mu, Sigma) / max_j p(X[t+1:,:]| q[t]=j, A, Mu, Sigma)
        (that's its definition.  That is not the way you should compute it).
    '''
    nframes, nstates = B.shape
    Beta_Hat = np.zeros(shape=(nframes, nstates))
    C = np.zeros(shape=nframes)
    # initialize
    Beta_Hat[nframes-1, :] = 1.0
    # iterate
    for t in range(nframes-2, -1, -1):
        for i in range(nstates):
            for j in range(nstates):
                Beta_Hat[t, i] += A[i, j] * B[t+1, j] * Beta_Hat[t+1, j]
        C[t] = max(Beta_Hat[t, :])
        Beta_Hat[t, :] /= C[t]
    return Beta_Hat

    raise NotImplementedError('You need to write this part!')

def posteriors(A, B, Alpha_Hat, Beta_Hat):
    '''Calculate the state and segment posteriors for an HMM.

    Inputs:
    A (nstates,nstates):
        A[y][i,j] = p(state[t]=j | state[t-1]=i)
    B (nframes,nstates):
        B[t,i] = p(X[t,:] | Mu[i,:], Sigma[i,:,:])
    Alpha_Hat (nframes,nstates):
        Alpha_Hat[t,i] = p(q=i | X[:t,:], A, Mu, Sigma)
    Beta_Hat (nframes,nstates):
        Beta_Hat[t,i] = p(X[t+1:,:]| q[t]=i, A, Mu, Sigma) / prod(G[t+1:])

    Returns:
    Gamma (nframes,nstates):
        Gamma[t,i] = p(q[t]=i | X, A, Mu, Sigma)
                   = Alpha_Hat[t,i]*Beta_Hat[t,i] / sum_i numerator
    Xi (nframes-1,nstates,nstates):
        Xi[t,i,j] = p(q[t]=i, q[t+1]=j | X, A, Mu, Sigma)
                  = Alpha_Hat[t,i]*A{i,j]*B[t+1,j]*Beta_Hat[t+1,j] / sum_{i,j} numerator

    
    Implementation Warning:
    The denominators, in either Gamma or Xi, might sometimes become 0 because of roundoff error.
    YOU MUST CHECK FOR THIS!
    Only perform the division if the denominator is > 0.
    If the denominator is == 0, then don't perform the division.
    '''
    nframes, nstates = B.shape
    Gamma = np.zeros(shape=(nframes, nstates))
    Xi = np.zeros(shape=(nframes-1, nstates, nstates))
    for t in range(nframes):
        for i in range(nstates):
            deno = np.sum(Alpha_Hat[t, :] * Beta_Hat[t, :])
            Gamma[t, i] = (Alpha_Hat[t, i] * Beta_Hat[t, i] / deno) if deno != 0.0 else 0.0

            if t < nframes - 1:
                for j in range(nstates):
                    deno = 0.0
                    for l in range(nstates):
                        deno += np.sum(Alpha_Hat[t, :] * A[:, l] * B[t+1, l] * Beta_Hat[t+1, l])
                    Xi[t, i, j] = (Alpha_Hat[t, i] * A[i, j] * B[t+1, j] * Beta_Hat[t+1, j] / deno) if deno != 0.0 else 0.0
    return Gamma, Xi

    raise NotImplementedError('You need to write this part!')

def E_step(X, Gamma, Xi):
    '''Calculate the expectations for an HMM.

    Inputs:
    X (nframes,nceps):
        X[t,:] = feature vector, t'th frame of n'th waveform
    Gamma (nframes,nstates):
        Gamma[t,i] = p(q[t]=i | X, A, Mu, Sigma)
    Xi (nsegments,nstates,nstates):
        Xi_list[t,i,j] = p(q[t]=i, q[t+1]=j | X, A, Mu, Sigma)
        WARNING: rows of Xi may not be time-synchronized with the rows of Gamma.  

    Returns:
    A_num (nstates,nstates): 
        A_num[i,j] = E[# times q[t]=i,q[t+1]=j]
    A_den (nstates): 
        A_den[i] = E[# times q[t]=i]
    Mu_num (nstates,nceps): 
        Mu_num[i,:] = E[X[t,:]|q[t]=i] * E[# times q[t]=i]
    Mu_den (nstates): 
        Mu_den[i] = E[# times q[t]=i]
    Sigma_num (nstates,nceps,nceps): 
        Sigma_num[i,:,:] = E[(X[t,:]-Mu[i,:])@(X[t,:]-Mu[i,:]).T|q[t]=i] * E[# times q[t]=i]
    Sigma_den (nstates): 
        Sigma_den[i] = E[# times q[t]=i]
    '''
    nframes, nstates = Gamma.shape
    nceps = X.shape[1]
    A_num = np.zeros(shape=(nstates, nstates))
    A_den = np.zeros(nstates)
    Mu_num = np.zeros(shape=(nstates, nceps))
    Mu_den = np.zeros(shape=(nstates))
    Mu = np.zeros(shape=(nstates, nceps))
    Sigma_num = np.zeros(shape=(nstates, nceps, nceps))
    Sigma_den = np.zeros(shape=(nstates))

    for i in range(nstates):
        for j in range(nstates):
            A_num[i, j] = np.sum(Xi[:, i, j])
            A_den[i] = np.sum(Xi[:, i, :])
        Mu_num[i, :] = np.sum(X * Gamma[:, i].reshape(-1, 1), axis=0)
        Mu_den[i] = np.sum(Gamma[:, i])
        Mu[i, :] = Mu_num[i, :] / Mu_den[i]
        Sigma_num[i, :, :] = np.dot((X - Mu[i, :]).T, (X - Mu[i, :]) * Gamma[:, i].reshape(-1, 1))
        Sigma_den[i] = Mu_den[i]
    return A_num, A_den, Mu_num, Mu_den, Sigma_num, Sigma_den


    raise NotImplementedError('You need to write this part!')

def M_step(A_num, A_den, Mu_num, Mu_den, Sigma_num, Sigma_den, regularizer):
    '''Perform the M-step for an HMM.

    Inputs:
    A_num (nstates,nstates): 
        A_num[i,j] = E[# times q[t]=i,q[t+1]=j]
    A_den (nstates): 
        A_den[i] = E[# times q[t]=i]
    Mu_num (nstates,nceps): 
        Mu_num[i,:] = E[X[t,:]|q[t]=i] * E[# times q[t]=i]
    Mu_den (nstates): 
        Mu_den[i] = E[# times q[t]=i]
    Sigma_num (nstates,nceps,nceps): 
        Sigma_num[i,:,:] = E[(X[t,:]-Mu[i,:])@(X[t,:]-Mu[i,:]).T|q[t]=i] * E[# times q[t]=i]
    Sigma_den (nstates): 
        Sigma_den[i] = E[# times q[t]=i]
    regularizer (scalar):
        Coefficient used for Tikohonov regularization of each covariance matrix.

    Returns:
    A (nstates,nstates):
        A[y][i,j] = p(state[t]=j | state[t-1]=i), estimated as
        E[# times q[t]=j and q[t-1]=i]/E[# times q[t-1]=i)].
    Mu (nstates,nceps):
        Mu[i,:] = mean vector of the i'th state, estimated as
        E[average of the frames for which q[t]=i].
    Sigma (nstates,nceps,nceps):
        Sigma[i,:,:] = covariance matrix, i'th state, estimated as
        E[biased sample covariance of the frames for which q[t]=i] + regularizer*I
    '''
    nstates, nceps = Mu_num.shape
    A = np.zeros_like(A_num)
    Mu = np.zeros_like(Mu_num)
    Sigma = np.zeros_like(Sigma_num)
    A = A_num / A_den.reshape(-1, 1)
    Mu = Mu_num / Mu_den.reshape(-1, 1)
    Sigma = Sigma_num / Sigma_den.reshape(-1, 1, 1) + regularizer * np.eye(nceps)

    return A, Mu, Sigma
    
    raise NotImplementedError('You need to write this part!')

def myrecognize(X, Models):
    '''Perform isolated-word speech recognition using trained Gaussian HMMs.

    Inputs:
    X (list of (nframes[n],nceps) arrays):
        X[n][t,:] = feature vector, t'th frame of n'th waveform
    Models (dict of tuples):
        Models[y] = (A, Mu, Sigma) for class y
        A (nstates,nstates):
             A[i,j] = p(state[t]=j | state[t-1]=i, Y=y).
        Mu (nstates,nceps):
             Mu[i,:] = mean vector of the i'th state for class y
        Sigma (nstates,nceps,nceps):
             Sigma[i,:,:] = covariance matrix, i'th state for class y

    Returns:
    logprob (dict of numpy arrays):
       logprob[y][n] = log p(X[n] | Models[y] )
    Y_hat (list of strings):
       Y_hat[n] = argmax_y p(X[n] | Models[y] )

    Implementation Hint: 
    For each y, for each n,
    call observation_pdf, then scaled_forward, then np.log, then np.sum.
    '''
    N = len(X)
    logprob = {}
    Y_hat = []
    labels = list(Models.keys())
    for y in labels:
        logprob[y] = np.zeros(N)
        for n in range(N):
            B = observation_pdf(X[n], Models[y][1], Models[y][2])
            Alpha_Hat, G = scaled_forward(Models[y][0], B)
            logprob[y][n] = np.sum(np.log(G))
    for n in range(N):
        Y_hat.append(labels[np.argmax([logprob[y][n] for y in labels])])
    return logprob, Y_hat