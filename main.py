import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal

audio_file_path = "000e260a.wav"

y, sr = librosa.load(audio_file_path, mono=False)

y_L = y[0]  # left channel
y_R = y[1]  # right channel

y_tp = y.transpose()

nfft = len(y_tp) * 8

fft_result_L = np.fft.fft(y_L, n=nfft, axis=0)
fft_result_R = np.fft.fft(y_R, n=nfft, axis=0)

# corr = np.correlate(fft_result_L, fft_result_R, "same")
corr = signal.correlate(fft_result_L, fft_result_R, "same", "direct")

result_mag = (2 / nfft) * np.abs(corr[0 : int(len(corr) / 2) + 1])

result_mag_norm = (result_mag - np.min(result_mag)) / (
    np.max(result_mag) - np.min(result_mag)
)

for index in range(len(result_mag_norm)):
    result_mag_norm[index] = 1 - result_mag_norm[index]

f_axis = np.linspace(0, sr / 2, len(result_mag_norm))

plt.figure(num=("IACC Graph"))
plt.semilogx(f_axis, result_mag_norm)
plt.grid()
plt.grid(which="minor", color="0.9")
plt.title("IACC for given file")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()
