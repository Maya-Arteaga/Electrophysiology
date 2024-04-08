#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:46:34 2024

@author: juanpablomayaarteaga
"""

import pandas as pd
import seaborn as sns
import functions as fx
import numpy as np
import matplotlib.pyplot as plt



# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Practica2/B/"
o_path = fx.set_path(i_path + "Output/")
raster_square_path= fx.set_path(o_path + "Square_plots/")
z_score_path= fx.set_path(o_path + "Z_Score_plots/")
sub_class_path=fx.set_path(o_path + "Sub_class_plots/")



# Path of files
file_paths = [
    f'{i_path}TiemposNeuS1.csv',  
    f'{i_path}TiemposNeuS1A.csv',
    f'{i_path}TiemposNeuS1B.csv',
    f'{i_path}TiemposNeuS1C.csv',  
    f'{i_path}TiemposNeuS1D.csv',
    f'{i_path}TiemposNeuS1E.csv',
    f'{i_path}TiemposNeuS1F.csv',  
    f'{i_path}TiemposNeuS1G.csv',
    f'{i_path}TiemposNeuS1H.csv',
    f'{i_path}TiemposNeuS1I.csv',  

]




# Assign to each file a variable


data_S1 = fx.read_file_action_potentials(file_paths[0])
data_S1A = fx.read_file_action_potentials(file_paths[1])
data_S1B = fx.read_file_action_potentials(file_paths[2])
data_S1C = fx.read_file_action_potentials(file_paths[3])
data_S1D = fx.read_file_action_potentials(file_paths[4])
data_S1E= fx.read_file_action_potentials(file_paths[5])
data_S1F = fx.read_file_action_potentials(file_paths[6])
data_S1G = fx.read_file_action_potentials(file_paths[7])
data_S1H = fx.read_file_action_potentials(file_paths[8])
data_S1I = fx.read_file_action_potentials(file_paths[9])




window=50
step=10

data= ["", "A", "B", "C", "D", "E", "F", "G", "H", "I"]

for file in data:
    fx.firing_raster_plot(data=eval(f"data_S1{file}"), 
                        window_size_ms=window, step_size_ms=step, 
                        path=raster_square_path, save_as=f'S1{file}')


for file in data:
    fx.z_firing_raster_plot(data=eval(f"data_S1{file}"), 
                        window_size_ms=window, step_size_ms=step, 
                        path=z_score_path, save_as=f'S1{file}')
    
    
    
    
    
datasets = {
    "": data_S1,
    "A": data_S1A,
    "B": data_S1B,
    "C": data_S1C,
    "D": data_S1D,
    "E": data_S1E,
    "F": data_S1F,
    "G": data_S1G,
    "H": data_S1H,
    "I": data_S1I
}

# Dictionary to store flattened data
flattened_data = {}

# Loop through the datasets and flatten each
for suffix, data in datasets.items():
    # Create a flattened version of each dataset
    flattened_data[f"S1{suffix}"] = np.concatenate([np.concatenate(trial) for trial in data])





fx.subplots_z_firing_rate_from_dictionary(flattened_data, 
                                        window_size_ms=window, step_size_ms=step, 
                                        path=z_score_path, title="Times S1", save_as=f'S1{file}',
                                        raws=3, columns=4)







fx.concatenate_z_score_firing_rate(flattened_data, 
                                        window_size_ms=window, step_size_ms=step, 
                                        path=z_score_path, save_as='S1')


