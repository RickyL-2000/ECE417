# %%
import os, h5py, importlib, submitted
import numpy as np
with h5py.File('data.hdf5','r') as f:
    print(f.keys())

# %%
train_waveforms = {}
dev_waveforms = {}
test_waveforms = {}
with h5py.File('data.hdf5','r') as f:
    for y in ['1','2','3']:
        train_waveforms[y] = [ f['train'][y][i][:] for i in sorted(f['train'][y].keys()) ]
        dev_waveforms[y] = [ f['dev'][y][i][:] for i in sorted(f['dev'][y].keys()) ]
        test_waveforms[y] = [ f['test'][y][i][:] for i in sorted(f['test'][y].keys()) ]
for y in train_waveforms.keys():
    print('Training data for class Y=s%s includes %d waveforms'%(y,len(train_waveforms[y])))
nframes = 1+int((10504-200)/80)
frames = np.array([train_waveforms['1'][0][t*80:t*80+200] for t in range(nframes)])
spectrogram = np.log(np.maximum(0.1,np.absolute(np.fft.fft(frames)[:,1:100])))
windowed_cepstrum = np.fft.fft(spectrogram)[:,0:25]
liftered_spectrum = np.real(np.fft.ifft(windowed_cepstrum))
windowed_cepstrum = np.real(windowed_cepstrum)

train_cepstra, train_spectra = submitted.compute_features(train_waveforms)
dev_cepstra, dev_spectra = submitted.compute_features(dev_waveforms)
test_cepstra, test_spectra = submitted.compute_features(test_waveforms)

# %%
importlib.reload(submitted)
nstates = 5
Lambda = {
    '1': submitted.initialize_hmm(train_cepstra['1'], nstates),
    '2': submitted.initialize_hmm(train_cepstra['2'], nstates),
    '3': submitted.initialize_hmm(train_cepstra['3'], nstates)
}
for y in Lambda.keys():
    print('\nThe A matrix for class "%s" is \n'%(y), Lambda[y][0])

# %%
B_dict = { y:[] for y in ['1','2','3'] }
for y in ['1','2','3']:
    for X in train_cepstra[y]:
        B = submitted.observation_pdf(X, Lambda[y][1], Lambda[y][2])
        B_dict[y].append(B)
    print('B_dict[%s] is a list of %d B matrices'%(y,len(B_dict[y])))

# %%
Alpha_dict = { y:[] for y in ['1','2','3'] }
G_dict = { y:[] for y in ['1','2','3'] }
for y in ['1','2','3']:
    for B in B_dict[y]:
        Alpha_Hat, G = submitted.scaled_forward(Lambda[y][0], B)
        Alpha_dict[y].append(Alpha_Hat)
        G_dict[y].append(G)
    print('Alpha_dict[%s] is a list of %d Alpha_Hat matrices'%(y,len(Alpha_dict[y])))

# %%
Beta_dict = { y:[] for y in ['1','2','3'] }
for y in ['1','2','3']:
    for B in B_dict[y]:
        Beta_Hat = submitted.scaled_backward(Lambda[y][0], B)
        Beta_dict[y].append(Beta_Hat)
    print('Beta_dict[%s] is a list of %d Beta_Hat matrices'%(y,len(Beta_dict[y])))

# %%
Gamma_dict = { y:[] for y in ['1','2','3'] }
Xi_dict = { y:[] for y in ['1','2','3'] }
for y in ['1','2','3']:
    for n, (B, Alpha_Hat, Beta_Hat) in enumerate(zip(B_dict[y], Alpha_dict[y], Beta_dict[y])):
        Gamma, Xi = submitted.posteriors(Lambda[y][0], B, Alpha_Hat, Beta_Hat)
        Gamma_dict[y].append(Gamma)
        Xi_dict[y].append(Xi)
    print('Gamma_dict[%s] is a list of %d Gamma matrices'%(y,len(Gamma_dict[y])))

# %%
expectations = {}
for y in ['1','2','3']:
    expectations[y] = submitted.E_step(np.concatenate(train_cepstra[y]), np.concatenate(Gamma_dict[y]), np.concatenate(Xi_dict[y]))
    print('expectations[%s] is a tuple with %d elements'%(y,len(expectations[y])))

# %%
nstates = 5
Lambda_new = {
    '1': submitted.M_step(*expectations['1'], 1),
    '2': submitted.M_step(*expectations['2'], 1),
    '3': submitted.M_step(*expectations['3'], 1)
}
for y in Lambda.keys():
    print('\nThe new A matrix for class "%s" is \n'%(y), Lambda_new[y][0])

# %%
logprob_dict = {}
Yhat_dict = {}
for y in ['1','2','3']:
    logprob_dict[y], Yhat_dict[y] =submitted.recognize(dev_cepstra[y], Lambda_new)
    print('For true label %s, the recognized labels are: '%(y), Yhat_dict[y])
