import wave
import numpy
import sys


def loadFiles(path):
    print "reading files..."
    
    files = [ f for f in listdir(path) if isfile(join(path,f)) and splitext(f)[1] == ".wav" ]

    samples = []
    maleCount = 0
    femaleCount = 0
    for f in files:
        p = path + '/' + f
        
        print "...", f
        with open(p, "rb") as wavFile:
            wavFile.read(24)
            rate = wavFile.read(4)
            rate = struct.unpack("<i",rate)[0]
            print "......rate: ", rate
            wavFile.read(6)
            bps = wavFile.read(2)
            bps = struct.unpack("<h",bps)[0]
            print "......bps: ", bps
            
            wavFile.read(8)
            print "......reading data"
            sig = []
            sampleSize = bps/8
            b = wavFile.read(int(sampleSize))
            while b != "":
                b = struct.unpack("<h", b)
                sig.append(b[0])
                b = wavFile.read(int(sampleSize))
                samples.append({'name': f, 'nameGender': f[-5:-4], 'signal': sig, 'sampleRate': rate})
                if f[-5:-4] == "M":
                    maleCount += 1
                else:
                    femaleCount += 1
                    counters = {"maleCount":maleCount, "femaleCount":femaleCount}
    return samples, counters

if __name__ == '__main__':
    loadFiles(sys.argv[1])
