#!/usr/bin/env python
# -*- coding: utf -*-

from scipy.io.wavfile import read,write
from pylab import plot,show,subplot,specgram
from pylab import *
from numpy import *
from scipy import *
from scikits.audiolab import wavread

#wczytujemy plik wav
data, sample_frequency,encoding = wavread("train/002_M.wav")
fig = gcf()
fig.canvas.set_window_title("Gender Recognition")

#deklarujemy zmienną sub, którą pyplot potrzebuje do robienia subplotów, ta zmienna działa tak:
#pierwsza cyfra - liczba supblotów
#druga cyfra - numer kolumny, w której umieszczamy subplot
#trzecia cyfra - numer wiersza, w której umieszczamy subplot
#na początku dałem 211 (na wypadek jeśli będzie jeden kanał), jeśli będzie ich więcej to zapisujemy nową wartość do subplota
sub = 211
#jeśli jest jeden kanał to data jest ciągiem intów, jeśli nie to data jest ciągiem tablic, dlatego taki warunek
if(type(data[0]) != float64):
	sub += ((len(data[0]) + 1) * 100)

#wyświetlamy pierwszy subplot (go wyświetlamy zawsze) z nałożonymi na siebie kanałami
subplot(sub)
ylabel("all channels")
plot(range(len(data)), data)
axhline(linewidth = 1, color = 'k', linestyle = '--')
#liczymy dataAv tutaj, bo spektogram dla dwóch kanałów nie ogarnia
dataAv = [mean(d) for d in data]

#jeśli jest więcej kanałów to wyświetlamy też inne supbloty, dla jednego kanału nie ma to sensu, bo wszystkie wykresy wyglądałyby tak samo
if(type(data[0]) != float64):
	#tworzymy drugi subplot ze średnią arytmetyczną z kanałów
	sub += 1
	subplot(sub)
	ylabel("channels' mean")
	plot(range(len(dataAv)), dataAv)
	axhline(linewidth = 1, color = 'k', linestyle = '--')

	#tworzymy trzeci wykres z pierwszym kanałem
	sub += 1
	subplot(sub)
	i = 1
	ylabel(str(i) + ". channel")
	dataL = [d[0] for d in data]
	plot(range(len(dataL)), dataL)
	axhline(linewidth = 1, color = 'k', linestyle = '--')	

	#tworzymy czwarty wykres z drugim kanałem
	sub += 1
	i += 1
	subplot(sub)
	ylabel(str(i) + ". channel")
	dataR = [d[1] for d in data]
	plot(range(len(dataR)), dataR)
	axhline(linewidth = 1, color = 'k', linestyle = '--')

#wyświetlamy spektogram
sub += 1
subplot(sub)
ylabel("spectogram")
Pxx, freqs, bins, im = specgram(dataAv, Fs = sample_frequency, NFFT = 1024, noverlap = 900, cmap=cm.gist_heat)

show()
