import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = 'Cowboy Bebop The Real Folk - cut.wav'

# waveform
plt.figure(figsize=(10, 5))
signal, sr = librosa.load(file, sr=22050)  # sample rate * T -> 22050 * 30 secounds
librosa.display.waveplot(signal, sr=sr, alpha=0.4)
# plt.rcParams['agg.path.chunksize'] = 100000
# plt.plot(signal[::100], sr[::100])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.show()

# fft -> spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))
left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]

plt.plot(left_frequency, left_magnitude, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()

# stft -> spectrogram
n_fft = 2048
hop_length = 512

# calculate duration hop length and window in seconds
hop_length_duration = float(hop_length)/sr
n_fft_duration = float(n_fft)/sr

print("STFT hop length duration is: {}s".format(hop_length_duration))
print("STFT window duration is: {}s".format(n_fft_duration))


stft = librosa.core.stft(signal,  n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)  # convert do Db the spectrogram

librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.show()

# MFCCS
MFFCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFFCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.title("MFCCs")
plt.show()
