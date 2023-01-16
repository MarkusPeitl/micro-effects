import argparse
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
import numpy as np

def plot_ffts(data_buffers, sample_rate, bit_depth):
    
    for data_buffer in data_buffers:
        plot_fft(data_buffer, sample_rate)

def plot_fft(data_buffer, sample_rate, bit_depth):
    channels_data = []
    if (len(data_buffer.shape) < 2):
        channels_data = [np.array(data_buffer, dtype=float)]
    else:
        for channel_data in data_buffer:
            channels_data.append(np.array(channel_data, dtype=float))

    bit_size = 2**bit_depth

    normalized_data = np.array(data_buffer, dtype=float)/bit_size
    #fourier_result = fft(normalized_data)
    fourier_result = np.fft.fft(normalized_data)
    #freq = np.fft.fftfreq(len(normalized_data))
    #print("frequency: " + str(freq))
    half_symmetric_result = fourier_result[:int(len(fourier_result)/2)]

    plt.plot(abs(half_symmetric_result),'r')
    plt.show()

def play_audio(buffer, sample_rate):
    import IPython.display as ipd
    ipd.Audio(buffer, rate=sample_rate)

def low_pass(souce_file_path, upper_bound_hz):

    sample_rate, data = wavfile.read(souce_file_path)
    print("Read file with sample rate: " + str(sample_rate) + " and shape " + str(data.shape))
    print("Sample rate is {:.1f} kHz".format(sample_rate/1000.0))
    print("Length in seconds: " + str(int(data.shape[0]/sample_rate)))

    if (len(data.shape) < 2):
        plot_fft(data, sample_rate, 32)
    else:
        plot_ffts(data, sample_rate, 32)
 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run markdown contained code"
    )
    parser.add_argument('source', help="Source of the documentation file")
    parser.add_argument('frequency', type=int, help="Source of the documentation file")
    arguments = parser.parse_args()
    source = arguments.source

    low_pass(source, arguments.frequency)
