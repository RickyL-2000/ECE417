'''
If you finish this module, you can submit it for extra credit.
'''
import numpy as np
import math

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
    raise RuntimeError("You need to implement this!")
