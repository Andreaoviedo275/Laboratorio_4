import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks 
from scipy.stats import ttest_1samp

# Función para crear un filtro Butterworth
def butter_filter(lowcut, highcut, fs, order=4, filter_type="bandpass"):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if filter_type == "bandpass":
        b, a = butter(order, [low, high], btype="bandpass")
    elif filter_type == "lowpass":
        b, a = butter(order, high, btype="lowpass")
    elif filter_type == "highpass":
        b, a = butter(order, low, btype="highpass")
    return b, a

# Cargar el archivo CSV
archivo_csv = "emg_Andre.csv"  # Cambia esto por el nombre de tu archivo
df = pd.read_csv(archivo_csv, header=None)
df.columns = ["Señal EMG"]

# Asignar la señal EMG a una variable llamada 'informacion'
informacion = df["Señal EMG"]

# Parámetros de la señal
fs = 1000  # Frecuencia de muestreo en Hz
lowcut = 20.0  # Umbral para el filtro pasa-altas (20 Hz)
highcut = 450.0  # Umbral para el filtro pasa-bajas (450 Hz)

# Mostrar en consola los parámetros
tiempo_muestreo = 1 / fs  # Tiempo de muestreo (segundos)
longitud_senal = len(df)

# Imprimir los parámetros en la consola
print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Tiempo de muestreo: {tiempo_muestreo:.6f} s")
print(f"Longitud de la señal: {longitud_senal} muestras")

# Graficar la señal EMG original (todo el conjunto de datos)
plt.figure(figsize=(15, 6))
plt.plot(df["Señal EMG"], linestyle="-", color="blue")
plt.xlabel("Muestras")
plt.ylabel("Amplitud EMG")
plt.title("Señal EMG Original")
plt.grid(True)
plt.show()

# Sección ampliada en el eje X (Zoom en una parte de la señal)
zoom_inicio, zoom_fin = 17000, 40000  # Ajusta estos valores según lo necesites
fragmento_emg = df["Señal EMG"][zoom_inicio:zoom_fin]

# Graficar la señal EMG fragmentada
plt.figure(figsize=(15, 6))
plt.plot(fragmento_emg, linestyle="-", color="green")
plt.xlabel("Muestras")
plt.ylabel("Amplitud EMG")
plt.title("Señal EMG Fragmentada")
plt.grid(True)
plt.show()

# **Aplicar filtro pasa altas (para eliminar ruido de baja frecuencia)**
b, a = butter_filter(lowcut, highcut, fs, filter_type="highpass")
fragmento_filtrado_pasa_altas = filtfilt(b, a, fragmento_emg)

# **Aplicar filtro pasa bajas (para eliminar ruido de alta frecuencia)**
b, a = butter_filter(lowcut, highcut, fs, filter_type="lowpass")
fragmento_filtrado_pasa_bajas = filtfilt(b, a, fragmento_emg)

# Graficar la señal filtrada con ambos filtros
plt.figure(figsize=(15, 6))

# Señal filtrada con filtro pasa altas
plt.subplot(3, 1, 1)
plt.plot(fragmento_filtrado_pasa_altas, label="Filtrada Pasa Altas", color='orange')
plt.title(f"Señal Filtrada Pasa Altas (>{lowcut} Hz)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid()

# Señal filtrada con filtro pasa bajas
plt.subplot(3, 1, 2)
plt.plot(fragmento_filtrado_pasa_bajas, label="Filtrada Pasa Bajas", color='red')
plt.title(f"Señal Filtrada Pasa Bajas (<{highcut} Hz)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid()

plt.tight_layout()
plt.show()

# Detectar contracciones (picos) usando find_peaks en la señal pasa-altas
max_valor = np.max(fragmento_filtrado_pasa_altas)
umbral_dinamico = 0.15 * max_valor  # 15% del valor máximo, ajustable

# Ajustar la distancia entre los picos (p. ej., 500 muestras entre contracciones)
distance = 500  # Ajusta según la longitud de las contracciones

# Detectar contracciones (picos) usando find_peaks
peaks, _ = find_peaks(fragmento_filtrado_pasa_altas, height=umbral_dinamico, distance=distance)  # Usamos el umbral dinámico

# Mostrar cantidad de contracciones detectadas en la consola
cantidad_contracciones_detectadas = len(peaks)
print(f"Contracciones detectadas: {cantidad_contracciones_detectadas}")

# Definir el fragmento que se ampliará
inicio = 0 # Muestra de inicio
fin = 3000    # Muestra de fin
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 1')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()

# Parámetros de la señal
fs = 1000  # Frecuencia de muestreo en Hz
lowcut = 20.0  # Umbral para el filtro pasa-altas (20 Hz)
highcut = 450.0  # Umbral para el filtro pasa-bajas (450 Hz)

# Definir el fragmento que se ampliará
inicio = 0  # Muestra de inicio
fin = 3000  # Muestra de fin
datos_ampliados = informacion[inicio:fin]

# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 1: {frecuencia_media:.2f} Hz")



inicio = 2300 # Muestra de inicio
fin = 6000 # Muestra de fin
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 2')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 2: {frecuencia_media:.2f} Hz")




inicio =5000#Muestra de inicio
fin =8500
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 3')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 3: {frecuencia_media:.2f} Hz")



inicio =7800
fin =11000
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 4')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()

# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 4: {frecuencia_media:.2f} Hz")




inicio =10200
fin =13000
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 5')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 5: {frecuencia_media:.2f} Hz")






inicio =12500
fin =16500
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 6')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 6: {frecuencia_media:.2f} Hz")



inicio =16000
fin =18500
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 7')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 7: {frecuencia_media:.2f} Hz")




inicio =18100
fin =21000
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 8')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 8: {frecuencia_media:.2f} Hz")




inicio =20800
fin =23800
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 9')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 9: {frecuencia_media:.2f} Hz")




inicio =23000
fin =26000
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 10')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 10:{frecuencia_media:.2f} Hz")



inicio =26000
fin =29000
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 11')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 11:{frecuencia_media:.2f} Hz")

inicio =28500
fin =31000
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 12')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 12:{frecuencia_media:.2f} Hz")



inicio =30500
fin =34000
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 13')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 13:{frecuencia_media:.2f} Hz")




inicio =33500
fin =37000
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 14')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 14:{frecuencia_media:.2f} Hz")




inicio =36000
fin =39500
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 15')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 15:{frecuencia_media:.2f} Hz")




inicio =39000
fin =40000
datos_ampliados = informacion[inicio:fin]
# Graficar el fragmento ampliado
plt.figure(figsize=(12, 5))
plt.plot(datos_ampliados, color='r')
plt.title('Contracción 16')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()
# Aplicar filtro pasa-altas para eliminar frecuencias de baja frecuencia (20 Hz)
b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
datos_filtrados = filtfilt(b, a, datos_ampliados)

# Normalizar la señal filtrada
datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))

# Realizar la Transformada de Fourier (FFT)
N = len(datos_normalizados)  # Número de puntos en la señal
fft_values = np.abs(fft(datos_normalizados))[:N // 2]  # Obtener magnitud (mitad positiva)
freqs = fftfreq(N, 1/fs)[:N // 2]  # Obtener frecuencias correspondientes

# Graficar el espectro de frecuencia con la FFT
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_values, color='b')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
plt.grid()
plt.show()

frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
print(f"Frecuencia media contracción 16:{frecuencia_media:.2f} Hz")



# Lista de frecuencias de las contracciones
freq_means = [
    52.88, 53.47, 52.49, 52.95, 50.67, 52.47, 52.80, 55.24,
    48.27, 49.30, 48.15, 74.76, 51.85, 51.38, 50.02, 51.47
]

# Valor de referencia para la hipótesis nula
ref_value = 50  # Por ejemplo, supongamos que estamos evaluando si la media es 50 Hz

# Test de hipótesis: prueba t de una muestra
t_stat, p_value = ttest_1samp(freq_means, ref_value)

print(f" Valor p: {p_value:.4f}")

# Evaluación del resultado
if p_value < 0.05:
 print(" Se rechaza la hipótesis nula.")
else:
 print(" No se rechaza la hipótesis nula. No hay suficiente evidencia.")
