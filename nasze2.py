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
    windowed = signal * hanning(len(signal))

    from pylab import subplot, plot, log, copy, show

    #harmonic product spectrum:
    c = abs(rfft(windowed))**2
    x = 1
    #maxharms = 16
    #for x in range(1,maxharms):
        #a = copy(c[::x]) #Should average or maximum instead of decimating
        ## max(c[::x],c[1::x],c[2::x],...)
        #c = c[:len(a)]
    i = argmax(abs(c))

    print 'Pass %d: %f Hz' % (x, fs * i / len(windowed))



#wczytujemy plik wav
data, sample_frequency, encoding = wavread("m1.wav")
print data
freq_from_HPS(data,sample_frequency)
print 'sf: %f' % sample_frequency
print 'encoding: %s' % encoding

w = 20           # częstotliwość próbkowania [Hz]
T = 2           # rozważany okres [s]

n = len(data)        # liczba próbek
T = len(data) / float(sample_frequency)
t = linspace(0, T, n, endpoint=False)  # punkty na osi OX [s]

#print T
#freq = numpy.fft.fftfreq(numpy.arange(len(data)).shape[-1])[:len(data)]
#freq = freq * sample_frequency / 1000  # make the frequency scale
#print len(freq)
#print len(data)



ft = rfft(data*hanning(len(data)))
mgft = abs(ft)
xVals = np.fft.rfftfreq(len(data), 1/sample_frequency)
subplot(221)
plot(abs(xVals[:len(mgft)]), mgft)
temp = data
# Window the data
#temp = temp * hamming(50)

# Transform with the FFT, Return Power
#freq_pwr  = 10*log10(1e-20+abs(rfft(temp)))

#print freq_pwr


#print mean(mgft)

#print fft

w = np.fft.fft(data)
ws = w # [:(len(w)/2)]
freqs =  np.fft.fftfreq(len(ws))
#print 'koperek!'
#print(freqs.min(),freqs.max())
# (-0.5, 0.499975)

# Find the peak in the coefficients
idx=np.argmax(np.abs(ws))
freq=freqs[idx]
freq_in_hertz=abs(freq*sample_frequency)
print ' frequency new:  %f' %  freq_in_hertz
# 439.8975


fft = scipy.fft(data)
print len(fft)
subplot(222)
plot(t,data)
subplot(223)
plot(t,fft)
subplot(224)
Pxx, freqs, bins, im = specgram(data, Fs=sample_frequency, NFFT=1024,
noverlap=900, cmap=cm.gist_heat)
show()
