import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Cargar la canción completa y el snippet de la canción
cancion_completa_path = "D:\Palmore\Semestre 6\Reconocimiento de patrones\Parcial V2\cancion2.wav"
cancion_completa_path2 = "D:\Palmore\Semestre 6\Reconocimiento de patrones\Parcial V2\cancion1.wav"
cancion_completa_path3 = "D:\Palmore\Semestre 6\Reconocimiento de patrones\Parcial V2\cancion3.wav"
snippet_path = "D:\Palmore\Semestre 6\Reconocimiento de patrones\Parcial V2\snippet1.wav"

cancion_completa, sr = librosa.load(cancion_completa_path, sr=None, mono=True, offset=0.0, duration=10.0)
cancion_completa2, sr = librosa.load(cancion_completa_path2, sr=None, mono=True, offset=0.0, duration=10.0)
cancion_completa3, sr = librosa.load(cancion_completa_path3, sr=None, mono=True, offset=0.0, duration=10.0)
snippet, sr = librosa.load(snippet_path, sr=None, mono=True, offset=0.0, duration=10.0)

# Calcular la huella acústica de la canción completa y del snippet
cancion_completa_fft = np.fft.fft(cancion_completa)
cancion_completa_fft_mag = np.abs(cancion_completa_fft)
cancion_completa_fft_freq = np.fft.fftfreq(len(cancion_completa)) * sr

cancion_completa_fft2 = np.fft.fft(cancion_completa2)
cancion_completa_fft_mag2 = np.abs(cancion_completa_fft2)
cancion_completa_fft_freq2 = np.fft.fftfreq(len(cancion_completa2)) * sr

cancion_completa_fft3 = np.fft.fft(cancion_completa3)
cancion_completa_fft_mag3 = np.abs(cancion_completa_fft3)
cancion_completa_fft_freq3 = np.fft.fftfreq(len(cancion_completa3)) * sr

snippet_fft = np.fft.fft(snippet)
snippet_fft_mag = np.abs(snippet_fft)
snippet_fft_freq = np.fft.fftfreq(len(snippet)) * sr

# Identificar los picos más fuertes de la huella acústica de la canción completa y del snippet
cancion_completa_peaks = argrelextrema(cancion_completa_fft_mag, np.greater)
cancion_completa_peaks2 = argrelextrema(cancion_completa_fft_mag2, np.greater)
cancion_completa_peaks3 = argrelextrema(cancion_completa_fft_mag3, np.greater)
snippet_peaks = argrelextrema(snippet_fft_mag, np.greater)

# Comparar los picos de la huella acústica de la canción completa y del snippet
coincidencias = 0
coincidencias2 = 0
coincidencias3 = 0
for peak in snippet_peaks[0]:
    if np.any(np.abs(cancion_completa_peaks[0] - peak) < .5):
        coincidencias += 1
        print("Se añadió una coincidencia a la canción 1.")
    if np.any(np.abs(cancion_completa_peaks2[0] - peak) < .5):
        coincidencias2 += 1
        print("Se añadió una coincidencia a la canción 2.")
    if np.any(np.abs(cancion_completa_peaks3[0] - peak) < .5):
        coincidencias3 += 1
        print("Se añadió una coincidencia a la canción 3.")

print("Coincidencias con la canción 1: ", coincidencias)
print("Coincidencias con la canción 2: ", coincidencias2)
print("Coincidencias con la canción 3: ", coincidencias3)

# Imprimir el resultado
if coincidencias > 100:
    print("La canción es la misma que la numero 1.")
elif coincidencias2 > 100:
    print("La canción es la misma que la numero 2 .") 
elif coincidencias3 > 100:
    print("La canción es la misma que la numero 3.")
else:
    print("Ninguna canción coincide con el snippet.")
