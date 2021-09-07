'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
import math

def make_frames(signal, hop_length, win_length):
    '''
    frames = make_frames(signal, hop_length, win_length)

    signal (num_samps) - the speech signal
    hop_length (scalar) - the hop length, in samples
    win_length (scalar) - the window length, in samples
    frames (num_frames, win_length) - array with one frame per row

    num_frames should be enough so that each sample from the signal occurs in at least one frame.
    The last frame may be zero-padded.
    '''
    num_frames = int(signal.shape[0] / hop_length)
    frames = np.zeros(shape=(num_frames, win_length))
    for i in range(num_frames):
        frames[i][: min(win_length, signal.shape[0] - i * hop_length)] = \
            signal[i * hop_length: min(i * hop_length + win_length, signal.shape[0])]
    return frames

def correlate(frames):
    '''
    autocor = correlate(frames)

    frames (num_frames, win_length) - array with one frame per row
    autocor (num_frames, 2*win_length-1) - each row is the autocorrelation of one frame
    '''
    num_frames, win_length = frames.shape
    autocor = np.zeros(shape=(num_frames, 2 * win_length - 1))
    for i in range(num_frames):
        autocor[i] = np.correlate(frames[i], frames[i], mode='full')
    return autocor

def make_matrices(autocor, p):
    '''
    R, gamma = make_matrices(autocor, p)

    autocor (num_frames, 2*win_length-1) - each row is symmetric autocorrelation of one frame
    p (scalar) - the desired size of the autocorrelation matrices
    R (num_frames, p, p) - p-by-p Toeplitz autocor matrix of each frame, with R[0] on main diagonal
    gamma (num_frames, p) - length-p autocor vector of each frame, R[1] through R[p]
    '''
    num_frames, win_length = autocor.shape
    win_length = (win_length + 1) // 2
    R = np.zeros(shape=(num_frames, p, p))
    gamma = np.zeros(shape=(num_frames, p))
    for i in range(num_frames):
        for j in range(p):
            R[i, j] = autocor[i, win_length - 1 - j: win_length - 1 + p - j]
        gamma[i] = autocor[i, win_length: win_length + p]
    return R, gamma

def lpc(R, gamma):
    '''
    a = lpc(R, gamma)
    Calculate the LPC coefficients in each frame

    R (num_frames, p, p) - p-by-p Toeplitz autocor matrix of each frame, with R[0] on main diagonal
    gamma (num_frames, p) - length-p autocor vector of each frame, R[1] through R[p]
    a (num_frames,p) - LPC predictor coefficients in each frame
    '''
    num_frames, p = gamma.shape
    a = np.zeros(shape=(num_frames, p))
    for i in range(num_frames):
        a[i] = np.linalg.inv(R[i]).dot(gamma[i])
    return a

def framepitch(autocor, Fs):
    '''
    framepitch = framepitch(autocor, samplerate)

    autocor (num_frames, 2*win_length-1) - autocorrelation of each frame
    Fs (scalar) - sampling frequency
    framepitch (num_frames) - estimated pitch period, in samples, for each frame, or 0 if unvoiced

    framepitch[t] = 0 if the t'th frame is unvoiced
    framepitch[t] = pitch period, in samples, if the t'th frame is voiced.
    Pitch period should maximize R[framepitch]/R[0], in the range 4ms <= framepitch < 13ms.
    Call the frame voiced if and only if R[framepitch]/R[0] >= 0.3, else unvoiced.
    '''
    num_frames, win_length = autocor.shape
    win_length = (win_length + 1) // 2
    framepitch = np.zeros(shape=num_frames)
    for i in range(num_frames):
        period = np.argmax(autocor[i, int(0.004 * Fs) + win_length - 1: 
                                            int(0.013 * Fs) + win_length - 1] /
                                            autocor[i, win_length - 1]) + int(0.004 * Fs)
        framepitch[i] = period \
            if autocor[i, period + win_length - 1] / autocor[i, win_length - 1] >= 0.3 \
            else 0.0
    return framepitch
            
def framelevel(frames):
    '''
    framelevel = framelevel(frames)

    frames (num_frames, win_length) - array with one frame per row
    framelevel (num_frames) - framelevel[t] = power (energy/duration) of the t'th frame, in decibels
    '''
    num_frames, win_length = frames.shape
    framelevel = np.zeros(shape=num_frames)
    for i in range(num_frames):
        framelevel[i] = 10 * np.log10(np.sum(frames[i] * frames[i]) / win_length)
    return framelevel

def interpolate(framelevel, framepitch, hop_length):
    '''
    samplelevel, samplepitch = interpolate(framelevel, framepitch, hop_length)

    framelevel (num_frames) - levels[t] = power (energy/duration) of the t'th frame, in decibels
    framepitch (num_frames) - estimated pitch period, in samples, for each frame, or 0 if unvoiced
    hop_length  (scalar) - number of samples between start of each frame
    samplelevel ((num_frames-1)*hop_length+1) - linear interpolation of framelevel
    samplepitch ((num_frames-1)*hop_length+1) - modified linear interpolation of framepitch

    samplelevel is exactly as given by numpy.interp.
    samplepitch is modified so that samplepitch[n]=0 if the current frame or next frame are unvoiced.
    '''
    num_frames = framelevel.shape[0]
    samplelevel = np.interp(np.linspace(0, num_frames - 1, (num_frames - 1) * hop_length + 1),
                            np.arange(num_frames), framelevel)
    samplepitch = np.interp(np.linspace(0, num_frames - 1, (num_frames - 1) * hop_length + 1),
                            np.arange(num_frames), framepitch)        
    for i in range(num_frames):
        if framepitch[i] == 0.0 or (i < num_frames-1 and framepitch[i+1] == 0.0):
            samplepitch[i * hop_length: (i + 1) * hop_length] = 0.0
    return samplelevel, samplepitch
    

def excitation(samplelevel, samplepitch):
    '''
    phase, excitation = excitation(samplelevel, samplepitch)

    samplelevel ((num_frames-1)*hop_length+1) - effective level (in dB) of every output sample
    samplepitch ((num_frames-1)*hop_length+1) - effective pitch period for every output sample
    phase ((num_frames-1)*hop_length+1) - phase of the fundamental at every output sample,
      modulo 2pi, so that 0 <= phase[n] < 2*np.pi for every n.
    excitation ((num_frames-1)*hop_length+1) - LPC excitation signal
      if samplepitch[n]==0, then excitation[n] is zero-mean Gaussian
      if samplepitch[n]!=0, then excitation[n] is a delta function time-aligned to the phase
      In either case, excitation is scaled so its average power matches samplelevel[n].
    '''
    ######## WARNING: the following lines must remain, so that your random numbers will match the grader
    from numpy.random import Generator, PCG64
    rg = Generator(PCG64(1234))
    ## Your Gaussian random numbers must be generated using the command ***rg.normal***
    ## (See https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html).
    ## (1) You must generate them in order, from the beginning to the end of the waveform.
    ## (2) You must generate a random sample _only_ if the corresponding samplepitch[n] > 0.
    length = samplelevel.shape[0]
    phase = np.zeros(shape=length)
    # excitation = (10 ** (samplelevel / 10)) ** 0.5
    excitation = np.zeros(shape=length)
    for i in range(length):
        if samplepitch[i] == 0:
            # if sample is unvoiced, don't increment the pitch phase
            phase[i] = phase[i-1] if i > 0 else 0.0
            excitation[i] = (10 ** (samplelevel[i] / 10)) ** 0.5 * rg.normal()
            # FIXME: why the instruction says must generate sample if pitch > 0? typo?
        elif samplepitch[i] > 0:
            phase[i] = phase[i-1] + 2 * np.pi / samplepitch[i] if i > 0 else 0.0
            if phase[i] >= 2 * np.pi:
                phase %= 2 * np.pi
                excitation[i] = (10 ** (samplelevel[i] / 10)) ** 0.5 * samplepitch[i] ** 0.5
            # else:
            #     excitation[i] = (10 ** (samplelevel[i] / 10)) ** 0.5 * rg.normal()
            # FIXME: according to the formula, this code above should exist. Why not?
    return phase, excitation

def synthesize(excitation, a):
    '''
    y = synthesize(excitation, a)
    excitation ((num_frames-1)*hop_length+1) - LPC excitation signal
    a (num_frames,p) - LPC predictor coefficients in each frame
    y ((num_frames-1)*hop_length+1) - LPC synthesized  speech signal
    '''
    length = excitation.shape[0]
    num_frames, p = a.shape
    hop_length = (length - 1) // (num_frames - 1)
    y = np.zeros(shape=length)
    for i in range(length):
        y[i] += excitation[i]
        for j in range(p):
            y[i] += a[i // hop_length, j] * (y[i - j - 1] if i - j - 1 >= 0 else 0.0)
    return y
