#!/usr/bin/env python
# -*- coding: utf -*-
from __future__ import division
from pylab import *
from numpy import *
from scipy import *

w = 20           # częstotliwość próbkowania [Hz]
T = 2            # rozważany okres [s]

n = T * w        # liczba próbek
t = linspace(0, T, n, endpoint=False) # punkty na osi OX [s]

f = lambda t : sin(2*pi*t)    # def. funkcji
signal = f(t)                 # funkcja spróbkowana
print signal 
subplot(211)
plot(t, signal, '*')

signal1 = fft(signal)      
signal1 = abs(signal1)        # moduł 

subplot(212)
freqs = range(n)              # <-- ZACZNIJ TUTAJ. Użyj linspace
stem(freqs, signal1, '-*')

show()
