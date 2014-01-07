#!/usr/bin/env python
# -*- coding: utf -*-
from __future__ import division
import numpy
from matplotlib.mlab import find
from time import time
import sys
from pylab import plot, show, subplot, specgram
from pylab import *
from numpy import *
from scipy import *
import math

import scipy
from scikits.audiolab import wavread
from scipy.signal import blackmanharris, fftconvolve
#from parabolic import parabolic
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, diff, log
from parabolic import parabolic
    #  from scipy.signal import blackmanharris, fftconvolve
def Pitch(signal):
    signal = np.fromstring(signal, 'Int16');
    crossing = [math.copysign(1.0, s) for s in signal]
    index = find(np.diff(crossing));
    f0=round(len(index) *sample_frequency /(2*np.prod(len(signal))))
    return f0;

def freq_from_fft(sig, fs):
    """Estimate frequency from peak of FFT
    """

    f = rfft(data)
    subplot (224)
    plot (f)

    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(f)) # Just use this for less-accurate, naive version
    print 'i: %f' % i
    print 'f[i]: %d' % f[i]
    true_i = parabolic(log(abs(f)), i)[0]
    print true_i
    # Convert to equivalent frequency

    return fs * true_i
def freq_from_HPS(signal, fs):
    """
    Estimate frequency using harmonic product spectrum (HPS)
    """
    windowed = signal * blackmanharris(len(signal))

    from pylab import subplot, plot, log, copy, show

    #harmonic product spectrum:
    c = abs(rfft(windowed))
    maxharms = 8
    subplot(maxharms,1,1)
    plot(log(c))
    for x in range(2,maxharms):
        a = copy(c[::x]) #Should average or maximum instead of decimating
        # max(c[::x],c[1::x],c[2::x],...)
        c = c[:len(a)]
        i = argmax(abs(c))
        true_i = parabolic(abs(c), i)[0]
        print 'Pass %d: %f Hz' % (x, fs * true_i / len(windowed))
        c *= a
        subplot(maxharms,1,x)
        plot(log(c))
    show()

def freq_from_autocorr(sig, fs):
    """Estimate frequency using autocorrelation
    """
    # Calculate autocorrelation (same thing as convolution, but with
    # one input reversed in time), and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)/2:]
    # Find the first low point
    d = diff(corr)
    start = find(d > 0)[0]
    # Find the next peak after the low point (other than 0 lag). This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    return fs / px

#wczytujemy plik wav
data, sample_frequency, encoding = wavread("f4.wav")

#print 'freq: %f' % freq_from_autocorr(data,sample_frequency)
print 'freq: %f' % Pitch(data)
print 'sf: %f' % sample_frequency
print 'encoding: %s' % encoding

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


ft = rfft(data * np.hanning(len(data)))
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
print 'koperek!'
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
