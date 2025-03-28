import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
from nidaqmx.constants import TerminalConfiguration
import time

# Configuración de la tarjeta DAQ
device_name = "Dev4"  # Nombre del dispositivo DAQ, usualmente 'Dev1'
emg_channel = "ai0"  # Canal de entrada para la señal EMG (canal analógico 0)
sampling_rate = 1000 # Frecuencia de muestreo en Hz
capture_time = 10  # Tiempo de captura en segundos (60 segundos = 1 minuto)

# Número de muestras que vamos a capturar
samples_to_read = sampling_rate * capture_time

# Crear un Task en el DAQ
with nidaqmx.Task() as task:
    # Configuración del canal de entrada (ai0 para EMG)
    task.ai_channels.add_ai_voltage_chan(f"{device_name}/{emg_channel}",
                                         terminal_config=TerminalConfiguration.RSE,
                                         min_val=-10, max_val=10)  # Rango de voltaje, puedes ajustarlo según la señal
    
    # Configuración de la adquisición (muestras por canal, frecuencia de muestreo)
    task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                    samps_per_chan=samples_to_read)

    # Iniciar la adquisición de datos
    print(f"Capturando la señal EMG durante {capture_time} segundos...")

    # Crear una lista vacía para almacenar los datos
    emg_data = []

    # Comenzamos la captura
    

    # Calculamos el tiempo de finalización
    end_time = time.time() + capture_time
    
    
    while len(emg_data)< 4*samples_to_read:
        # Leer las muestras disponibles
        task.start()
        data = task.read(samples_to_read)  # Leer las muestras definidas
        emg_data.extend(data)  # Añadir los datos leídos a la lista emg_data
        print(len(emg_data))
        task.stop()

        
    # Detener el task después de la captura
    
    # Convertir los datos leídos a un arreglo numpy
    emg_data = np.array(emg_data)
    np.savetxt("emg_Andre.csv", emg_data,delimiter=" ")
    # Verificar que los datos han sido leídos correctamente
    if emg_data.size > 0:
        print(f"Tipo de datos: {type(emg_data)}")
        print(f"Primeros 10 datos: {emg_data[:10]}")  # Ver los primeros 10 valores

        # Calcular el tiempo correspondiente para cada muestra
        time_axis = np.linspace(0, 6*capture_time, len(emg_data))  # Tiempo desde 0 hasta el tiempo de captura

        # Graficar los datos
        plt.figure(figsize=(20, 6))
        plt.plot(time_axis, emg_data)  # Usamos time_axis como el eje X
        plt.title(f"Señal EMG capturada durante {capture_time} segundos")
        plt.xlabel("Tiempo (segundos)")
        plt.ylabel("Voltaje (V)")
        plt.grid(True)
        plt.show()
    else:
        print("Error: No se capturaron datos. Verifique la configuración.")
        
