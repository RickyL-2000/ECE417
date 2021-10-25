'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
import math

def smooth_video(x, sigma, L):
    '''
    y = smooth_video(x, sigma, L)
    Smooth the video using a sampled-Gaussian smoothing kernel.

    x (TxRxC) - a video with T frames, R rows, C columns
    sigma (scalar) - standard deviation of the Gaussian smoothing kernel
    L (scalar) - length of the Gaussian smoothing kernel
    y (TxRxC) - the same video, smoothed in the row and column directions.
    '''
    kernel = np.zeros(shape=L)
    for i in range(L):
        kernel[i] = np.exp(- 0.5 * ((i - (L-1)/2) / sigma) ** 2) / (2 * np.pi * sigma ** 2) ** 0.5
    # print(kernel)
    y = np.zeros(shape=x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[i, j, :] = np.convolve(x[i, j, :], kernel, 'same')
        for j in range(x.shape[2]):
            y[i, :, j] = np.convolve(y[i, :, j], kernel, 'same')
    return y

def gradients(x):
    '''
    gt, gr, gc = gradients(x)
    Compute gradients using a first-order central finite difference.

    x (TxRxC) - a video with T frames, R rows, C columns
    gt (TxRxC) - gradient in the time direction
    gr (TxRxC) - gradient in the vertical direction
    gc (TxRxC) - gradient in the horizontal direction
    '''
    gt = np.zeros(shape=x.shape)
    gr = np.zeros(shape=x.shape)
    gc = np.zeros(shape=x.shape)

    kernel = np.array([0.5, 0.0, -0.5])

    T, R, C = x.shape
    for t in range(T):
        for r in range(R):
            gc[t, r, :] = np.convolve(x[t, r, :], kernel, 'same')
        for c in range(C):
            gr[t, :, c] = np.convolve(x[t, :, c], kernel, 'same')
    for r in range(R):
        for c in range(C):
            gt[:, r, c] = np.convolve(x[:, r, c], kernel, 'same')

    gt[0, :, :] = 0
    gt[-1, :, :] = 0
    gr[:, 0, :] = 0
    gr[:, -1, :] = 0
    gc[:, :, 0] = 0
    gc[:, :, -1] = 0
    return gt, gr, gc

def lucas_kanade(gt, gr, gc, H, W):
    '''
    vr, vc = lucas_kanade(gt, gr, blocksize)

    gt (TxRxC) - gradient in the time direction
    gr (TxRxC) - gradient in the vertical direction
    gc (TxRxC) - gradient in the horizontal direction
    H (scalar) - height (in rows) of each optical flow block
    W (scalar) - width (in columns) of each optical flow block
    
    Within each HxW block of each frame, you should create:
     - b vector, of size (H*W,1)
     - A matrix, of size (H*W,2)
     - calculate v = pinv(A)*b
     - assign vr and vc as the two elements of the v vector

    vr (Txint(R/H)xint(C/W)) - pixel velocity in vertical direction
    vc (Txint(R/H)xint(C/W)) - pixel velocity in horizontal direction
    '''
    T, R, C = gt.shape
    b = np.zeros(shape=(H*W, 1))
    A = np.zeros(shape=(H*W, 2))
    vr = np.zeros(shape=(T, int(R/H), int(C/W)))
    vc = np.zeros(shape=(T, int(R/H), int(C/W)))

    for t in range(gt.shape[0]):
        for i in range(int(R/H)):
            for j in range(int(C/W)):
                b = - gt[t, int(i*H): int((i+1)*H), int(j*W): int((j+1)*W)].reshape(H*W, 1)
                A[:, 0] = gc[t, int(i*H): int((i+1)*H), int(j*W): int((j+1)*W)].reshape(H*W)
                A[:, 1] = gr[t, int(i*H): int((i+1)*H), int(j*W): int((j+1)*W)].reshape(H*W)
                v = np.linalg.pinv(A).dot(b)
                vc[t, i, j], vr[t, i, j] = v[0], v[1]
    return vr, vc

def medianfilt(x, H, W):
    '''
    y = medianfilt(x, H, W)
    Median-filter the video, x, in HxW blocks.

    x (TxRxC) - a video with T frames, R rows, C columns
    H (scalar) - the height of median-filtering blocks
    W (scalar) - the width of median-filtering blocks
    y (TxRxC) - y[t,r,c] is the median of the pixels x[t,rmin:rmax,cmin:cmax], where
      rmin = max(0,r-int((H-1)/2))
      rmax = min(R,r+int((H-1)/2)+1)
      cmin = max(0,c-int((W-1)/2))
      cmax = min(C,c+int((W-1)/2)+1)
    '''
    T, R, C = x.shape
    y = np.zeros(shape=x.shape)
    for t in range(x.shape[0]):
        for r in range(x.shape[1]):
            for c in range(x.shape[2]):
                y[t, r, c] = np.median(x[t, max(0, r-int((H-1)/2)): min(R, r+int((H-1)/2)+1), 
                                            max(0, c-int((W-1)/2)): min(C, c+int((W-1)/2)+1)])
    return y
            
def interpolate(x, U):
    '''
    y = interpolate(x, U)
    Upsample and interpolate an image using bilinear interpolation.

    x (TxRxC) - a video with T frames, R rows, C columns
    U (scalar) - upsampling factor
    y (Tx(U*R)x(U*C)) - interpolated image
    '''
    T, R, C = x.shape
    y = np.zeros(shape=(T, U*R, U*C))

    for t in range(T):
        for r in range(R):
            y[t, U * r, :] = np.interp(x=np.arange(U*C), 
                                       xp=np.append(np.arange(0, U*C, U), U*C),     # not to U*C-1, must be U*C
                                       fp=np.append(x[t, r, :], 0.0))
        for c in range(C*U):
            y[t, :, c] = np.interp(x=np.arange(U*R), 
                                   xp=np.append(np.arange(0, U*R, U), U*R), 
                                   fp=np.append([y[t, i, c] for i in range(0, U*R, U)], 0.0))
    return y

def scale_velocities(v, U):
    '''
    delta = scale_velocities(v, U)
    Scale the velocities in v by a factor of U,
    then quantize them to the nearest integer.
    
    v (TxRxC) - T frames, each is an RxC velocity image
    U (scalar) - an upsampling factor
    delta (TxRxC) - integers closest to v*U
    '''
    return np.array(list(map(lambda x: int(np.round(x)), (v * U).reshape(-1)))).reshape(v.shape)

def velocity_fill(x, vr, vc, keep):
    '''
    y = velocity_fill(x, vr, vc, keep)
    Fill in missing frames by copying samples with a shift given by the velocity vector.

    x (T,R,C) - a video signal in which most frames are zero
    vr (T,Ra,Cb) - the vertical velocity field, integer-valued
    vc (T,Ra,Cb) - the horizontal velocity field, integer-valued
        Notice that Ra and Cb might be less than R and C.  If they are, the remaining samples 
        of y should just be copied from y[t-1,r,c].
    keep (array) -  a list of frames that should be kept.  Every frame not in this list is
     replaced by samples copied from the preceding frame.

    y (T,R,C) - a copy of x, with the missing frames filled in.
    '''
    T, R, C = x.shape
    T, Ra, Cb = vr.shape
    y = np.zeros(shape=x.shape)
    for t in range(T):
        if t in keep:
            y[t, :, :] = x[t, :, :]
            continue
        for r in range(R):
            for c in range(C):
                if r >= Ra or c >= Cb:
                    y[t, r, c] = y[t-1, r, c]
                else:
                    y[t, r, c] = y[t-1, 
                                   max(min(int(r - vr[t-1, r, c]), R-1), 0), 
                                   max(min(int(c - vc[t-1, r, c]), C-1), 0)]
    return y
