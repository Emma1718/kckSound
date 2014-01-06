#!/usr/bin/env python
# -*- coding: utf -*-
from __future__ import division
import numpy
from time import time
import sys
from pylab import plot, show, subplot, specgram
from pylab import *
from numpy import *
from scipy import *
import scipy
from scikits.audiolab import wavread
from scipy.signal import blackmanharris
#from parabolic import parabolic
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, diff, log
from parabolic import parabolic
    #  from scipy.signal import blackmanharris, fftconvolve

def freq_from_fft(sig, fs):
    """Estimate frequency from peak of FFT
    """

    f = rfft(data)
    subplot (224)
    plot (f)

    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(f)) # Just use this for less-accurate, naive version
    print 'i: %f' % i
    true_i = parabolic(log(abs(f)), i)[0]
    print true_i
    # Convert to equivalent frequency

    return fs * true_i

#wczytujemy plik wav
data, sample_frequency, encoding = wavread("m3.wav")

print 'freq: %f' % freq_from_fft(data,sample_frequency)


w = 20           # częstotliwość próbkowania [Hz]
T = 2           # rozważany okres [s]

n = len(data)        # liczba próbek
T = len(data) / float(sample_frequency)
t = linspace(0, T, n, endpoint=False)  # punkty na osi OX [s]

print T
freq = numpy.fft.fftfreq(numpy.arange(len(data)).shape[-1])[:len(data)]
freq = freq * sample_frequency / 1000  # make the frequency scale
print len(freq)
print len(data)


ft = np.fft.rfft(data * np.hanning(len(data)))
mgft = abs(ft)
xVals = np.fft.rfftfreq(len(data), 1 / float(sample_frequency))
subplot(221)
plot(abs(xVals[:len(mgft)]), mgft)
temp = data
# Window the data
#temp = temp * hamming(50)

# Transform with the FFT, Return Power
freq_pwr  = 10*log10(1e-20+abs(rfft(temp)))

print freq_pwr


print mean(mgft)

print fft

w = np.fft.fft(data)
freqs = np.fft.fftfreq(len(w))
print(freqs.min(),freqs.max())
# (-0.5, 0.499975)

# Find the peak in the coefficients
idx=np.argmax(np.abs(w)**2)
freq=freqs[idx]
freq_in_hertz=abs(freq*sample_frequency)
print 'frequency new: %f' % freq_in_hertz
# 439.8975


fft = scipy.fft(data)
print len(fft)
subplot(222)
plot(t,data)
subplot(223)
plot(t,fft)
#subplot(224)
#Pxx, freqs, bins, im = specgram(data, Fs=sample_frequency, NFFT=1024,
#noverlap=900, cmap=cm.gist_heat)
show()
