import pyaudio
import wave
import sys

import os
import pylab
import struct
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt

CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "f1.wav"


def recording_wav():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()



"""Generate a Spectrogram image for a given WAV audio sample.

A spectrogram, or sonogram, is a visual representation of the spectrum
of frequencies in a sound.  Horizontal axis represents time, Vertical axis
represents frequency, and color represents amplitude.
"""

def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig('spectrogram.png')
    pylab.clf();


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def power(wav_file):
    rate, data = wavfile.read(wav_file)
    t = np.arange(len(data[:,0]))  #  *1.0/rate
    pylab.plot(t, data[:,0])
    pylab.savefig('wave_left_channel.png')
    pylab.clf()

#To get the frequency and amplitude of you wave, do FFT.
#Following code plot the power of every frequency bin:

    p = 20*np.log10(np.abs(np.fft.rfft(data[:2048, 0])))

    f = np.linspace(0, rate/2.0, len(p))
    pylab.plot(f, p)
    pylab.xlabel("Frequency(Hz)")
    pylab.ylabel("Power(dB)")
    pylab.savefig('power.png')
    pylab.clf()

def recognition(WAVE_OUTPUT_FILENAME):
    # Read in a WAV and find the freq's

    chunk = CHUNK

    # open up a wave
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'rb')

    swidth = wf.getsampwidth()
    RATE = wf.getframerate()

    # use a Blackman window
    window = np.blackman(chunk)
    # open stream
    p = pyaudio.PyAudio()
    stream = p.open(format =
                    p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = RATE,
                    output = True)

    # read some data
    data = wf.readframes(chunk)
    print chunk*swidth
    print len(data)
    # play stream and find the frequency of each chunk
   # while len(data) == chunk*swidth:
    while data != '':
        # write data out to the audio stream
        stream.write(data)
        # unpack the data and times by the hamming window
        indata = np.array(wave.struct.unpack("%dh"%(len(data)/swidth),\
                                             data))#*window
        # Take the fft and square each value
        fftData=abs(np.fft.rfft(indata))**2
        # find the maximum
        which = fftData[1:].argmax() + 1
        # use quadratic interpolation around the max
        if which != len(fftData)-1:
            y0,y1,y2 = np.log(fftData[which-1:which+2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            # find the frequency and output it
            thefreq = (which+x1)*RATE/chunk
            print "The freq is %f Hz." % (thefreq)
        else:
            thefreq = which*RATE/chunk
            print "The freq is %f Hz." % (thefreq)
        # read some more data
        data = wf.readframes(chunk)
    if data:
        stream.write(data)
    stream.close()
    p.terminate()


if __name__ == '__main__':

    recording_wav()
    #graph_spectrogram(WAVE_OUTPUT_FILENAME)
    power(WAVE_OUTPUT_FILENAME)
    recognition(WAVE_OUTPUT_FILENAME)
