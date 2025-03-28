# Laboratorio_4

📌 Descripción

El análisis de la fatiga muscular es crucial en rehabilitación y ergonomía. Este laboratorio permite procesar señales EMG adquiridas de un músculo en contracción sostenida para observar cambios que indican fatiga.

🎯 Objetivos

✔ Aplicar técnicas de filtrado digital para limpiar la señal EMG.

✔ Detectar picos en la señal y evaluar su evolución en el tiempo.

✔ Calcular la frecuencia mediana para analizar la fatiga muscular.

✔ Visualizar los resultados a través de gráficos.

🛠 Instalación y Ejecución

1️⃣ Requisitos Previos
Debes tener instalado Python 3.x y las siguientes librerías:

        pip install numpy pandas scipy matplotlib
        
2️⃣ Ejecución
-  Coloca el archivo emg_Andre.csv en la misma carpeta que Lab 4.py.
-  Ejecuta el código en una terminal o entorno de desarrollo:
  
        python Lab\ 4.py

3️⃣ Las gráficas se mostrarán en pantalla y también se pueden guardar en la carpeta Figures/ (según se modifique el código).

🔎 Explicación del Código

El código se divide en varias secciones que cubren desde la carga de datos hasta la prueba de hipótesis.

1️⃣ Importación de Librerías

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.fft import fft, fftfreq
        from scipy.signal import butter, filtfilt, find_peaks 
        from scipy.stats import ttest_1samp


🔹 ¿Qué hace esto? Importa las librerías necesarias para procesamiento de señales y visualización.

- pandas y numpy: Para manejo y procesamiento de datos.

- matplotlib.pyplot: Para la generación de gráficos.

- scipy.fft y fftfreq: Para calcular la Transformada de Fourier y obtener las frecuencias.

- scipy.signal: Para aplicar filtros digitales y detectar picos.

- scipy.stats: Para realizar la prueba t.

2️⃣ Función para Crear un Filtro Butterworth

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

🔹 ¿Qué hace esto? Define un filtro digital para limpiar la señal EMG y reducir ruido.

Parámetros:

- lowcut y highcut: Frecuencias de corte para el filtro.

- fs: Frecuencia de muestreo.

- order: Orden del filtro.

- filter_type: Tipo de filtro ("bandpass", "lowpass" o "highpass").

3️⃣ Señal EMG Fragmentada

Para analizar mejor la señal, se toma un fragmento específico
        
        inicio = 17000
        fin = 40000
        fragmento_emg = df["Señal EMG"][inicio:fin]
        plt.figure(figsize=(15, 6))
        plt.plot(fragmento_emg, color='green')
        plt.title("Señal EMG Fragmentada")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud EMG")
        plt.grid(True)
        plt.show()

![Figure 2025-03-27 211345 (1)](https://github.com/user-attachments/assets/79f22794-8998-4fc2-bbda-6aed43690d05)

Fig 1. Señal Fragmentada

4️⃣ Carga y Visualización de la Señal Original

        archivo_csv = "emg_Andre.csv"  # Cambia esto por el nombre de tu archivo
        df = pd.read_csv(archivo_csv, header=None)
        df.columns = ["Señal EMG"]
        
        # Asignar la señal a una variable
        informacion = df["Señal EMG"]
        
        # Parámetros de la señal
        fs = 1000  # Frecuencia de muestreo (Hz)
        lowcut = 20.0  # Filtro pasa-altas (20 Hz)
        highcut = 450.0  # Filtro pasa-bajas (450 Hz)
        
        tiempo_muestreo = 1 / fs  # Tiempo entre muestras
        longitud_senal = len(df)
        
        print(f"Frecuencia de muestreo: {fs} Hz")
        print(f"Tiempo de muestreo: {tiempo_muestreo:.6f} s")
        print(f"Longitud de la señal: {longitud_senal} muestras")
        
        # Graficar la señal EMG original
        plt.figure(figsize=(15, 6))
        plt.plot(df["Señal EMG"], linestyle="-", color="blue")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud EMG")
        plt.title("Señal EMG Original")
        plt.grid(True)
        plt.show()

![Figure 2025-03-27 211345 (0)](https://github.com/user-attachments/assets/c3ede83f-ca42-4590-b333-2f843e28327e)

Fig 2. Señal EMG Original 

💡 Interpretación: Se carga el archivo CSV y se muestra la señal cruda.

5️⃣ Aplicación de Filtros a un Fragmento de la Señal

Nota: Se asume que previamente se ha definido un fragmento llamado fragmento_emg (puede ser un segmento específico de la señal).

5.1 Filtro Pasa-Altas y Pasa-Bajas

        # Aplicar filtro pasa-altas (elimina ruido de baja frecuencia)
        b, a = butter_filter(lowcut, highcut, fs, filter_type="highpass")
        fragmento_filtrado_pasa_altas = filtfilt(b, a, fragmento_emg)
        
        # Aplicar filtro pasa-bajas (elimina ruido de alta frecuencia)
        b, a = butter_filter(lowcut, highcut, fs, filter_type="lowpass")
        fragmento_filtrado_pasa_bajas = filtfilt(b, a, fragmento_emg)

💡 Interpretación: Se aplican dos filtros para comparar el efecto de cada uno.

![Figure 2025-03-27 211345 (2)](https://github.com/user-attachments/assets/c8e82b87-7de9-4860-bcec-787925233e83)

Fig 3. Filtrado pasa_bajas y pasa_altas

En este laboratorio, se definió una frecuencia de muestreo de 1000 Hz. De acuerdo con el teorema de Nyquist, la frecuencia máxima que puede capturarse sin aliasing es la mitad de la frecuencia de muestreo (en este caso, 500 Hz). Por ello, cualquier frecuencia por encima de 500 Hz no se representará correctamente en la señal digitalizada.

Para filtrar la señal EMG se ha seleccionado un rango de 20 Hz a 450 Hz:

20 Hz (pasa-altas): Elimina componentes de muy baja frecuencia o ruido de alta amplitud y baja frecuencia (por ejemplo, fluctuaciones de línea base).

450 Hz (pasa-bajas): Atenúa ruido de muy alta frecuencia, evitando acercarse demasiado a la frecuencia de Nyquist (500 Hz).

Este pasa-banda (20–450 Hz) abarca las frecuencias de interés en la mayoría de las señales EMG, ya que la mayor parte de la energía electromiográfica se encuentra por debajo de 500 Hz, con picos importantes en torno a 50–150 Hz, dependiendo del músculo y el tipo de contracción.

5.2 Graficación de las Señales Filtradas

        plt.figure(figsize=(15, 6))
        
        # Gráfica del filtro pasa-altas
        plt.subplot(3, 1, 1)
        plt.plot(fragmento_filtrado_pasa_altas, label="Filtrada Pasa Altas", color='orange')
        plt.title(f"Señal Filtrada Pasa Altas (>{lowcut} Hz)")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud")
        plt.grid()
        
        # Gráfica del filtro pasa-bajas
        plt.subplot(3, 1, 2)
        plt.plot(fragmento_filtrado_pasa_bajas, label="Filtrada Pasa Bajas", color='red')
        plt.title(f"Señal Filtrada Pasa Bajas (<{highcut} Hz)")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud")
        plt.grid()
        
        plt.tight_layout()
        plt.show()

💡 Interpretación: Se muestran ambas señales filtradas para visualizar el efecto de cada filtro.

6️⃣ Detección de Contracciones (Picos)

        max_valor = np.max(fragmento_filtrado_pasa_altas)
        umbral_dinamico = 0.15 * max_valor  # Umbral del 15% del valor máximo
        
        distance = 500  # Mínima distancia entre picos en muestras
        peaks, _ = find_peaks(fragmento_filtrado_pasa_altas, height=umbral_dinamico, distance=distance)
        cantidad_contracciones_detectadas = len(peaks)
        print(f"Contracciones detectadas: {cantidad_contracciones_detectadas}")

💡 Interpretación:

- Se utiliza un umbral dinámico para detectar picos, que representan contracciones musculares.

- Se ajusta la distancia mínima entre picos para evitar múltiples detecciones en una misma contracción.

7️⃣ Análisis Detallado de Contracciones

Para cada contracción se selecciona un segmento de la señal, se aplica el filtro, se normaliza y se realiza la FFT para calcular la frecuencia media. A continuación se muestran las secciones para cada contracción (en este ejemplo se analizan 16 contracciones).

Ejemplo para Contracción 1:

       # Selección del segmento (Contracción 1: muestras 0 a 3000)
        inicio = 0
        fin = 3000
        datos_ampliados = informacion[inicio:fin]
        
        # Graficar el fragmento ampliado
        plt.figure(figsize=(12, 5))
        plt.plot(datos_ampliados, color='r')
        plt.title('Contracción 1')
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.show()
        
        # Filtrado pasa-altas
        b, a = butter_filter(lowcut=lowcut, highcut=highcut, fs=fs, filter_type="highpass")
        datos_filtrados = filtfilt(b, a, datos_ampliados)
        
        # Normalización
        datos_normalizados = datos_filtrados / np.max(np.abs(datos_filtrados))
        
        # FFT
        N = len(datos_normalizados)
        fft_values = np.abs(fft(datos_normalizados))[:N // 2]
        freqs = fftfreq(N, 1/fs)[:N // 2]
        
        # Graficar espectro
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, fft_values, color='b')
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Magnitud")
        plt.title("Espectro de Frecuencia (FFT Filtrada y Normalizada)")
        plt.grid()
        plt.show()
        
        # Cálculo de la frecuencia media
        frecuencia_media = np.sum(freqs * fft_values) / np.sum(fft_values)
        print(f"Frecuencia media contracción 1: {frecuencia_media:.2f} Hz")

![Figure 2025-03-27 211345 (3)](https://github.com/user-attachments/assets/54702e7f-6125-4b49-b08b-1da49c08bc14)

Fig 4. Contracción 1

![Figure 2025-03-27 211345 (4)](https://github.com/user-attachments/assets/59bca8c9-0c64-4c53-a4a7-b5a817334a46)

Fig 5. Espectro de frecuencia de la contracción 1.

Análisis para Contracciones 2 a 16:

El mismo procedimiento se repite para distintos segmentos de la señal:

- Contracción 2: Muestras de 2300 a 6000
  
![Figure 2025-03-27 211345 (5)](https://github.com/user-attachments/assets/2a6330e6-e10f-4a53-a715-b0cbd87096fa)

Fig 6. Contracción 2

![Figure 2025-03-27 211345 (6)](https://github.com/user-attachments/assets/2b5b2f35-1006-4cad-aef6-cccbcce2dbc3)

Fig 7. Espectro de frecuencia de la contracción 2.

- Contracción 3: Muestras de 5000 a 8500

![Figure 2025-03-27 211345 (7)](https://github.com/user-attachments/assets/ccff369b-a1bf-4d36-9c41-fd94817716ae)

Fig 8. Contracción 3

![Figure 2025-03-27 211345 (8)](https://github.com/user-attachments/assets/b7d499ab-2c6c-409d-ab4d-1734cb798349)

Fig 9. Espectro de frecuencia de la contracción 3.

- Contracción 4: Muestras de 7800 a 11000

![Figure 2025-03-27 211345 (9)](https://github.com/user-attachments/assets/740336e7-b83c-407c-97d0-ed0d0d4b62be)

Fig 10. Contracción 4

![Figure 2025-03-27 211345 (10)](https://github.com/user-attachments/assets/c1fd7ec7-790a-4e1c-a00a-9c523d2e842e)

Fig 11. Espectro de frecuencia de la contracción 4.

- Contracción 5: Muestras de 10200 a 13000

![Figure 2025-03-27 211345 (11)](https://github.com/user-attachments/assets/bd74c7c1-3f9c-4540-96ac-a1da87eec76f)

Fig 12. Contracción 5

![Figure 2025-03-27 211345 (12)](https://github.com/user-attachments/assets/1a81350d-2cc8-4fdb-a837-d37a1d8bc65c)

Fig 13. Espectro de frecuencia de la contracción 5.

- Contracción 6: Muestras de 12500 a 16500

![Figure 2025-03-27 211345 (13)](https://github.com/user-attachments/assets/94a8e843-0671-46a8-adfc-f048c651e2a7)

Fig 14. Contracción 6

![Figure 2025-03-27 211345 (14)](https://github.com/user-attachments/assets/1cff0ac7-f676-4634-83a1-485e6a6422ed)

Fig 15. Espectro de frecuencia de la contracción 6.

- Contracción 7: Muestras de 16000 a 18500

![Figure 2025-03-27 211345 (15)](https://github.com/user-attachments/assets/e89bb87a-0ca9-48db-8548-b408b458381f)

Fig 16. Contracción 7

![Figure 2025-03-27 211345 (16)](https://github.com/user-attachments/assets/d5e81b81-c4bd-4c38-a877-ca6681ef806b)

Fig 17. Espectro de frecuencia de la contracción 7.

- Contracción 8: Muestras de 18100 a 21000

![Figure 2025-03-27 211345 (17)](https://github.com/user-attachments/assets/65150953-3a3c-4d72-a234-e8ac6f36f4d9)

Fig 18. Contracción 8

![Figure 2025-03-27 211345 (18)](https://github.com/user-attachments/assets/1b6b3d5e-ac51-4946-869c-6cdf671fc3bb)

Fig 19. Espectro de frecuencia de la contracción 8.

- Contracción 9: Muestras de 20800 a 23800
![Figure 2025-03-27 211345 (19)](https://github.com/user-attachments/assets/07d62bb7-a71a-4027-af08-dfbf3f32ef75)


Fig 20. Contracción 9

![Figure 2025-03-27 211345 (20)](https://github.com/user-attachments/assets/9731a7d0-0123-4895-bebf-06ca0274e509)

Fig 21. Espectro de frecuencia de la contracción 9.

- Contracción 10: Muestras de 23000 a 26000

![Figure 2025-03-27 211345 (21)](https://github.com/user-attachments/assets/b1abac35-a220-4472-a964-cbb8a841491b)

Fig 22. Contracción 10

![Figure 2025-03-27 211345 (22)](https://github.com/user-attachments/assets/f00463ca-ea7a-4d45-a7d1-c82c1350bfa0)

Fig 23. Espectro de frecuencia de la contracción 10.

- Contracción 11: Muestras de 26000 a 29000

![Figure 2025-03-27 211345 (23)](https://github.com/user-attachments/assets/fd0f6794-bc39-4649-8a68-7f82be5956c6)

Fig 24. Contracción 11

![Figure 2025-03-27 211345 (24)](https://github.com/user-attachments/assets/7e1ebb9f-8b0d-43c4-9e4a-1a33750f0ad8)
Fig 25. Espectro de frecuencia de la contracción 11.


- Contracción 12: Muestras de 28500 a 31000

![Figure 2025-03-27 211345 (25)](https://github.com/user-attachments/assets/8821efdc-3758-4740-b13f-5797f06e820a)

Fig 26. Contracción 12

![Figure 2025-03-27 211345 (26)](https://github.com/user-attachments/assets/a600f5f4-3adb-46a0-af38-1894586e9cd3)

Fig 27. Espectro de frecuencia de la contracción 12.

- Contracción 13: Muestras de 30500 a 34000

![Figure 2025-03-27 211345 (27)](https://github.com/user-attachments/assets/42f367b0-c97a-4751-9335-8edf875325a4)

Fig 28. Contracción 13

![Figure 2025-03-27 211345 (28)](https://github.com/user-attachments/assets/a92cd836-b1d0-4c3b-8e05-d0657092ef03)

Fig 29. Espectro de frecuencia de la contracción 13.

- Contracción 14: Muestras de 33500 a 37000

![Figure 2025-03-27 211345 (29)](https://github.com/user-attachments/assets/c3d23398-e814-4874-9ab7-c5ef066fd568)

Fig 30. Contracción 14

![Figure 2025-03-27 211345 (30)](https://github.com/user-attachments/assets/dccf2557-cce2-422c-93fc-be947609668a)

Fig 31. Espectro de frecuencia de la contracción 14.

- Contracción 15: Muestras de 36000 a 39500

![Figure 2025-03-27 211345 (31)](https://github.com/user-attachments/assets/0cc7dd47-dba0-4375-a4f9-755671a72cde)

Fig 32. Contracción 15

![Figure 2025-03-27 211345 (32)](https://github.com/user-attachments/assets/9b572361-0977-480b-ae22-0cc5cda4f021)

Fig 33. Espectro de frecuencia de la contracción 15.

- Contracción 16: Muestras de 39000 a 40000

![Figure 2025-03-27 211345 (33)](https://github.com/user-attachments/assets/4caa0d47-f262-486e-95a0-c8b4bec44615)

Fig 34. Contracción 16

![Figure 2025-03-27 211345 (34)](https://github.com/user-attachments/assets/7082ce28-4ac9-46d7-ba6c-a5e20a948762)

Fig 35. Espectro de frecuencia de la contracción 16.

8️⃣ Evaluación Estadística: Prueba t de Una Muestra

Una vez calculadas las frecuencias medias de las contracciones, se agrupan en una lista:

        freq_means = [
            52.88, 53.47, 52.49, 52.95, 50.67, 52.47, 52.80, 55.24,
            48.27, 49.30, 48.15, 74.76, 51.85, 51.38, 50.02, 51.47
        ]

- Valor de referencia: Se compara con 50 Hz (hipótesis nula: la media es 50 Hz).

        ref_value = 50
        t_stat, p_value = ttest_1samp(freq_means, ref_value)
        print(f" Valor p: {p_value:.4f}")
        
        if p_value < 0.05:
            print(" Se rechaza la hipótesis nula.")
        else:
            print(" No se rechaza la hipótesis nula. No hay suficiente evidencia.")

💡 Interpretación:

- Si p_value < 0.05, se rechaza la hipótesis nula, indicando que la media de las frecuencias es significativamente diferente de 50 Hz.

- Si p_value >= 0.05, no se rechaza la hipótesis nula.


9️⃣ Resultados en Consola

![image](https://github.com/user-attachments/assets/5aff78e1-0c9c-4917-9a46-9243b0c33d76)

Fig 36. Resultado de consola

![image](https://github.com/user-attachments/assets/df660922-97f4-4773-9fd6-f47546c41914)

Fig 37.Resultado de consola

📌 Conclusión
Este laboratorio demuestra cómo procesar señales EMG para evaluar la fatiga muscular. A través del filtrado digital, detección de picos, análisis espectral y pruebas estadísticas, se puede obtener información valiosa sobre la activación y fatiga muscular.

Mejoras posibles:

- Automatizar la segmentación de contracciones.

- Aplicar pruebas de normalidad antes de la prueba t.

- Graficar la evolución de la frecuencia media en un solo diagrama para visualizar tendencias.
