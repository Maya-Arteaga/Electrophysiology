#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:02:27 2024

@author: juanpablomayaarteaga
"""

"""

Hay tres archivos de datos que acompa ̃nan a esta pr ́actica: 
    TiemposNeuDPC1.csv,
    TiemposNeuDPC2.csv y 
    TiemposNeuS1.csv. 

Estos archivos contienen la informaci ́on de los
tiempos en que ocurrieron los -potenciales de acci ́on de una neurona-. 
Dicho registro fue realizado mientras un sujeto realizaba una -tarea de comparaci ́on de patrones temporales-
(presentada en la clase 1). Cada archivo tiene -cuatro bloques-, 
cada bloque representa una -condici ́on experimental o ”clase” distinta-.

Los bloques est ́an separados por una salto de l ́ınea (una l ́ınea sin nada). 
Cada l ́ınea con datos representa una repetici ́on de la tarea o un ”ensayo”. 

Las lineas son secuencias de n ́umeros separados por comas: cada n ́umero es un
tiempo. Los bloques est ́an formados por 15 ensayos, por lo que hay un total de 60 ensayos.



"""


import pandas as pd
import seaborn as sns
import functions as fx
import numpy as np
import matplotlib.pyplot as plt



# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Practica1/"
o_path = fx.set_path(i_path + "Output/")
raster_path= fx.set_path(o_path + "Raster_plots/")
bin_path= fx.set_path(o_path + "Bin_plots/")
square_path= fx.set_path(o_path + "Square_plots/")
gaussian_path= fx.set_path(o_path + "Gaussian_plots/")
alpha_path= fx.set_path(o_path + "Alpha_plots/")


# Path of files
file_paths = [
    f'{i_path}TiemposNeuDPC1.csv',  
    f'{i_path}TiemposNeuDPC2.csv',
    f'{i_path}TiemposNeuS1.csv'
]

# Assign to each file a variable
data1 = fx.read_file_action_potentials(file_paths[0])
data2 = fx.read_file_action_potentials(file_paths[1])
data3 = fx.read_file_action_potentials(file_paths[2])

data = [data1, data2, data3]

# Print the number of blocks and trials to know how is structured the data
data_names = ['data1', 'data2', 'data3']
for i, d in enumerate(data):
    print("")
    
    print(f"Number of blocks in {data_names[i]}: {len(d)}")
    # Show the number of trials in the first block
    print(f"Number of trials in the 1 block of {data_names[i]}: {len(d[0])}")
    print(f"Number of trials in the 2 block of {data_names[i]}: {len(d[1])}")
    print(f"Number of trials in the 3 block of {data_names[i]}: {len(d[2])}")
    print(f"Number of trials in the 4 block of {data_names[i]}: {len(d[3])}")
    
    print("")



#Raster Plot: a spike train frequencies during each essay
fx.raster_plot(data1, path=raster_path, save_as="TiemposNeuDPC1")
fx.raster_plot(data2, path=raster_path, save_as="TiemposNeuDPC2")
fx.raster_plot(data3, path=raster_path, save_as="TiemposNeuS1")






bin_width=[10, 50, 100, 200]

for width in bin_width:    
    fx.binning_firing_rate(data=data1, bin_width_ms=width, path=bin_path, save_as='data1')
    fx.binning_firing_rate(data=data1, bin_width_ms=width, path=bin_path, save_as='data2')
    fx.binning_firing_rate(data=data1, bin_width_ms=width, path=bin_path, save_as='data3')







window_sizes=[50, 200, 400]
step_sizes=[10, 50, 100]

#Sliding square window
for window in window_sizes:
    for step in step_sizes:
        fx.square_firing_rate(data=data1, window_size_ms=window, step_size_ms=step, path=square_path, save_as="data1")
        fx.square_firing_rate(data=data2, window_size_ms=window, step_size_ms=step, path=square_path, save_as="data2")
        fx.square_firing_rate(data=data3, window_size_ms=window, step_size_ms=step, path=square_path, save_as="data3")







window_sizes=[50, 400]
step_sizes=[50, 100]
sigma= [1, 50]

#Sliding Gaussian window
for window in window_sizes:
    for step in step_sizes:
        for s in sigma:
            fx.gaussian_firing_rate(data=data1, window_size_ms=window, step_size_ms=step, sigma=s, path=gaussian_path, save_as="data1")
            fx.gaussian_firing_rate(data=data1, window_size_ms=window, step_size_ms=step, sigma=s, path=gaussian_path, save_as="data2")
            fx.gaussian_firing_rate(data=data1, window_size_ms=window, step_size_ms=step, sigma=s, path=gaussian_path, save_as="data3")
    


fx.alpha_firing_rate(data=data1, window_size_ms=50, step_size_ms=50, alpha=10, path=alpha_path, save_as='data1')




