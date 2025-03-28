import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

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

# Parámetros de la señal
fs = 1000  # Frecuencia de muestreo en Hz
lowcut = 20.0  # Umbral para el filtro pasa-altas (20 Hz)
highcut = 450.0  # Umbral para el filtro pasa-bajas (450 Hz)

# Mostrar en consola los parámetros
tiempo_muestreo = 1 / fs  # Tiempo de muestreo (segundos)
longitud_senal = len(df)
musculo_medido = "Bíceps"  # Asumimos que el músculo medido es el bíceps (ajustar según se requiera)

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
# Ajustar el parámetro 'height' según la amplitud de tus contracciones
max_valor = np.max(fragmento_filtrado_pasa_altas)
umbral_dinamico = 0.15 * max_valor  # 15% del valor máximo, ajustable

# Ajustar la distancia entre los picos (p. ej., 500 muestras entre contracciones)
distance = 500  # Ajusta según la longitud de las contracciones

# Detectar contracciones (picos) usando find_peaks
peaks, _ = find_peaks(fragmento_filtrado_pasa_altas, height=umbral_dinamico, distance=distance)  # Usamos el umbral dinámico

# Mostrar cantidad de contracciones detectadas en la consola
cantidad_contracciones_detectadas = len(peaks)
print(f"Contracciones detectadas: {cantidad_contracciones_detectadas}")

# Sección ampliada en el eje X (Zoom en una parte de la señal)
zoom_inicio, zoom_fin = 0, 25000  # Ajusta estos valores según lo necesites
fragmento_emg = df["Señal EMG"][zoom_inicio:zoom_fin]

# Graficar la señal EMG fragmentada
plt.figure(figsize=(15, 6))
plt.plot(fragmento_emg, linestyle="-", color="green")
plt.xlabel("Muestras")
plt.ylabel("Amplitud EMG")
plt.title("Señal EMG Fragmentada")
plt.grid(True)
plt.show()

