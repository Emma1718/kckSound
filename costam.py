from Numeric import *
from MLab import *
from FFT import *
from dislin import *
import wave
import sys
import struct


# open the wave file
fp = wave.open(sys.argv[1],"rb")

sample_rate = fp.getframerate()
total_num_samps = fp.getnframes()
window_size = int(sys.argv[2])
num_fft = total_num_samps - window_size

# create temporary working array
temp = zeros((num_fft,window_size),Float)
signal = fp.readframes(total_num_samps)

# read in the data from the file
for i in range(num_fft):
    tempb = signal[i:i+window_size]#fp.readframes(window_size);
    temp[i,:] = array(struct.unpack("%dB"%(window_size),tempb),Float) - 128.0
fp.close()

# Window the data
temp = temp * hamming(window_size)

# Transform with the FFT, Return Power
freq_pwr  = 10*log10(1e-20+abs(real_fft(temp,fft_length)))

n_out_pts = (window_size / 2) + 1

# Plot the result
y_axis = 0.5*float(sample_rate) / n_out_pts * arange(n_out_pts)
x_axis = (total_num_samps / float(sample_rate)) / num_fft * arange(num_fft)
setvar('X',"Time (sec)")
setvar('Y',"Frequency (Hertz)")
conshade(freq_pwr,x_axis,y_axis)
disfin()
