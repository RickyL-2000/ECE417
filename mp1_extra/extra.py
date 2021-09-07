'''
If you finish this module, you can submit it for extra credit.
'''
import numpy as np
import math

"""
autocor only: 0.07
energy only: < 0.06

-_- so only using the energy (I even just randomly pick a threshold) can pass the test...
"""

def better_vad(signal, samplerate):
    '''
    vuv = better_vad(signal, samplerate)
    
    signal (sig_length) - a speech signal
    samplerate (scalar) - the sampling rate, samples/second
    vuv (sig_length) - vuv[n]=1 if signal[n] is  voiced, otherwise vuv[n]=0
    
    Write a function that decides whether each frame is voiced or not.
    You're provided with one labeled training example, and one labeled test example.
    You are free to use any external data you want.
    You can also use any algorithms from the internet that you want, 
    except that
    (1) Don't copy code.  If your code is similar to somebody else's, that's OK, but if it's the
    same, you will not get the extra credit.
    (2) Don't import any modules other than numpy and the standard library.
    '''
    # what if using just the default?
    hop_length = int(0.015 * samplerate)
    win_length = int(0.03 * samplerate)
    sig_length = signal.shape[0]
    vuv = np.zeros(shape=sig_length)

    frames = make_frames(signal, hop_length, win_length)
    # print(frames)
    # print("max =", np.max(frames))
    num_frames = frames.shape[0]

    vuv_using_autocor = vad_using_autocor(frames, samplerate, sig_length, hop_length)
    vuv_using_energy = vad_using_energy(frames, threshold=-60)

    vuv = vuv_using_energy
    return vuv

def vad_using_energy(frames, threshold):
    energy = np.mean(to_dB(frames ** 2), axis=1)
    # print(energy)
    energy[energy >= threshold] = 1.0
    energy[energy < threshold] = 0.0
    # print("energy =", energy)
    # print("max energy =", max(energy))
    # print("min energy =", min(energy))
    # print("mean energy =", np.mean(energy))
    return energy

def to_dB(array):
    array[array == 0.0] = 0.00000001
    ret = 10 * np.log10(array)
    ret[ret < -100] = -100
    return ret

def vad_using_autocor(frames, samplerate, sig_length, hop_length):
    num_frames = frames.shape[0]
    vuv = np.zeros(shape=sig_length)
    autocor = correlate(frames)
    framepitch = get_framepitch(autocor, samplerate)
    # print("framepitch =", framepitch)
    for i in range(num_frames):
        vuv[i * hop_length: min((i + 1) * hop_length, sig_length)] = 1.0 if framepitch[i] > 0 else 0.0
    return vuv


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

def get_framepitch(autocor, Fs):
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