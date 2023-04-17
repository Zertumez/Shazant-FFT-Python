import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

# Sampling frequency
frequency = 44100

# Recording duration in seconds
duration = 10

input("Presione Enter para comenzar a grabar...")
print("Grabando audio")

# grabar audio desde el micrófono a través de sound-device en una matriz Numpy
recording = sd.rec(int(duration * frequency), samplerate=frequency, channels=1)

# Esperar a que se complete la grabación
sd.wait()

print("Grabación finalizada")

# guardar la grabación en formato .wav utilizando scipy
write("D:\Palmore\Semestre 6\Reconocimiento de patrones\Parcial V2\snippet0.wav", frequency, recording)

# guardar la grabación en formato .wav utilizando wavio
wv.write("D:\Palmore\Semestre 6\Reconocimiento de patrones\Parcial V2\snippet0.wav", recording, frequency, sampwidth=2)


# Cargar la canción completa y el snippet de la canción
canciones = [
    {"nombre": "A tender feeling", "ruta": "D:\Palmore\Semestre 6\Reconocimiento de patrones\Parcial V2\cancion1.wav"},
    {"nombre": "Leave out all the rest", "ruta": "D:\Palmore\Semestre 6\Reconocimiento de patrones\Parcial V2\cancion2.wav"},
    {"nombre": "Shadow of the day", "ruta": "D:\Palmore\Semestre 6\Reconocimiento de patrones\Parcial V2\cancion3.wav"},
]
snippet_path = "D:\Palmore\Semestre 6\Reconocimiento de patrones\Parcial V2\snippet0.wav"

sr = None
duration = 10.0

canciones_fft = []
canciones_fft_mag = []
canciones_fft_freq = []
canciones_peaks = []

# Calcular la huella acústica de cada canción completa
for cancion in canciones:
    cancion_completa, sr = librosa.load(cancion["ruta"], sr=sr, mono=True, offset=0.0, duration=duration)
    cancion_fft = np.fft.fft(cancion_completa)
    cancion_fft_mag = np.abs(cancion_fft)
    cancion_fft_freq = np.fft.fftfreq(len(cancion_completa)) * sr
    cancion_peaks = argrelextrema(cancion_fft_mag, np.greater)
    
    canciones_fft.append(cancion_fft)
    canciones_fft_mag.append(cancion_fft_mag)
    canciones_fft_freq.append(cancion_fft_freq)
    canciones_peaks.append(cancion_peaks)

# Calcular la huella acústica del snippet
snippet, sr = librosa.load(snippet_path, sr=sr, mono=True, offset=0.0, duration=duration)
snippet_fft = np.fft.fft(snippet)
snippet_fft_mag = np.abs(snippet_fft)
snippet_fft_freq = np.fft.fftfreq(len(snippet)) * sr

# Identificar los picos más fuertes de la huella acústica del snippet
snippet_peaks = argrelextrema(snippet_fft_mag, np.greater)

# Comparar los picos de la huella acústica del snippet con los de cada canción completa
coincidencias = []
for idx, cancion in enumerate(canciones_peaks):
    num_coincidencias = 0
    for peak in snippet_peaks[0]:
        if np.any(np.abs(cancion[0] - peak) < 0.5):
            num_coincidencias += 1
    coincidencias.append(num_coincidencias)

# Imprimir el resultado
max_coincidencias = max(coincidencias)
if max_coincidencias > 100:
    indice_cancion = coincidencias.index(max_coincidencias)
    print(f"La canción es: {canciones[indice_cancion]['nombre']}.")
else:
    print("Ninguna canción coincide con el snippet.")
