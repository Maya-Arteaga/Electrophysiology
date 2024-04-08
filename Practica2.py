#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:14:02 2024

@author: juanpablomayaarteaga
"""

"""
Firing Rate: The Ice Cream Scoops Analogy
Imagine you're comparing how much ice cream two friends eat on average each day. 
Let's say Friend A eats 3 scoops a day, and Friend B eats 5 scoops a day. 
In this analogy, the "scoops of ice cream" represent the firing rate of neurons, 
or how many times a neuron "fires" or sends a signal in a given period. 
Just as you can say Friend B eats more ice cream on average, you can compare neurons or
 conditions based on their firing rates to see which has more activity.

Example in Neuroscience: If a neuron fires 20 times per second (20 Hz) 
when a light is on but only 5 times per second (5 Hz) in the dark, you might conclude that 
this neuron is more active in response to light. This is like observing that 
you eat more ice cream on hot days than cold ones.

Z-Score: The Ice Cream Contest Analogy
Now, imagine there's a contest to see who can eat the most ice cream compared to 
their usual amount. On the contest day, Friend A eats 5 scoops, and Friend B eats 7 scoops. 
To fairly compare them, you look at how much more each friend ate compared to their average. 
Friend A usually eats 3 scoops, so eating 5 is a significant increase. 
Friend B usually eats 5 scoops, so eating 7 is also an increase but not as significant 
proportionally as for Friend A. By calculating how much more each friend ate compared to
 their average (and considering their usual variation), you're doing something similar to
 calculating the z-score in neuroscience.

Example in Neuroscience: Using the z-score, you could determine how much more active a
 neuron is during a specific event compared to its normal activity level. 
 If a neuron usually fires 20 times per second but fires at 50 times per second when 
 a cat sees a mouse, you could calculate the z-score to see how significant 
 this spike in activity is relative to its normal variation. 
 This helps you understand if the change in firing rate is just within the normal 
 variability of the neuron or if it's a significant response to the mouse.


Simplified Insight
Firing Rate: Tells you how active something is in absolute terms, 
like counting scoops of ice cream to measure how much someone eats.

Z-Score: Helps you understand how unusual or significant a particular observation is
compared to the norm, like deciding who's the ice cream eating champion by considering
 both how much they ate and how much they usually eat.
 
 
In Conclusion
Firing rates give you a direct measure of activity (like counting scoops of ice cream), 
which is great for straightforward comparisons. Z-scores help you understand 
the context of that activity (like figuring out who really stepped up their ice cream 
                              game during a contest),
 providing insights into how unusual an activity level is compared to
 what's typical for that neuron.



PRACTICA:

Hay m ́ultiples archivos de datos que acompa ̃nan a esta pr ́actica. La mayor ́ıa de estos
archivos contienen la informaci ́on de los tiempos en que ocurrieron los potenciales de acci ́on
de una neurona. Dichos registros fueron realizados mientras un sujeto realizaba una tarea
de discriminaci ́on de patrones temporales (clase 1) o una tarea de detecci ́on de est ́ımulos
vibrot ́actiles. Las neuronas pueden ser de  ́areas corticales distintas: los archivos con “DPC”
en el nombre contienen datos de la corteza premotora dorsal (Dorsal Premotor Cortex); los
que tienen “S1” son de la corteza somatosensorial primaria (Primary Somatosensory Cortex).


Cada archivo relativo a la discriminaci ́on de patrones temporales, cuyos nombres
comienzan con “Tiempos”, tiene cuatro bloques: cada bloque representa una condici ́on
experimental o “clase” distinta. Los bloques est ́an separados por una salto de l ́ınea (una
l ́ınea sin nada). Cada l ́ınea con datos representa una repetici ́on de la tarea (un “ensayo”).

Los archivos con datos de la tarea de detecci ́on empiezan con “Det”. En esta tarea hab ́ıa
seis condiciones experimentales, cada una corresponde a un valor distinto de amplitud de
estimulaci ́on: 0μm (no hay est ́ımulo), 6μm, 8μm, 10μm, 12μm y 24μm. Similar a los de
discariminaci ́on, estos archivos est ́an divididos en bloques separados por un salto de l ́ınea.
Los bloques van de de menor a mayor: el primer bloque es de la condic ́on con 0μm y
el  ́ultimo es con 24μm. El n ́umero de ensayos en cada archivo no es necesariamente el mismo.



Recuerda que los tiempos de ocurrencia de las espigas est ́an en segundos y tienen como
referencia el momento en que comienza el primer est ́ımulo en ese ensayos: o sea, t = 0s es el
inicio del primer est ́ımulo. Tiempos negativos preceden la llegada del primer est ́ımulo.




"""


"""
1. (1 a tiempo; 0.8 fuera de tiempo) Este ejercicio es sobre la transformada z (z-scores),
y vamos a usar los siguientes archivos: de la tarea de discriminaci ́on de patrones
TiemposNeuDPC1.csv, TiemposNeuDPC2.csv y TiemposNeuS1.csv; de la tarea de
detecci ́on DetNeuS1 B y NeuDetS1 D. Antes de empezar, calcule las tasas de disparo
de las cinco neuronas: para las tres neuronas de S1 emplee ventanas de 50ms y pasos
de 10 ms; para las dos de DPC, ventanas de 200ms y pasos de 40ms. Vamos a calcular
la evoluci ́on temporal para diferentes z-scores promedio por clase y neurona. Para todos
los incisos, grafique cada una de sus curvas con una barra de error que indique ± una
desviaci ́on est ́andar del correspondiente z-score.


(a) Obten y grafica el z-score (para cada neurona y clase) usando la tasa de todas
las ventanas de tiempo (t ∈ [−2, 8] en la tarea de patrones temporales 
                        y t ∈ [−2, 3,5] para detecci ́on) 
y ensayos para calcular el valor medio y la desviaci ́on est ́andar con las
que se computa la transformada z en cada clase.


(b) Obten y grafica el z-score (para cada neurona y clase) usando la tasa de las ventanas
comprendidas entre -2s y 0s (a este tipo de tasa le llamamos tasa basal) y todos los
ensayos (k = 60) para calcular el valor medio y la desviaci ́on est ́andar con las que
se computa la transformada z en cada clase.



(c) Obten y grafica el z-score (para cada neurona y clase) usando la tasa de cada ventana
de tiempo particular y todos los ensayos para calcular el valor medio y la desviaci ́on
est ́andar con las que se computa la transformada z en cada tiempo y clase. Esto
es, para computar el z-score en el tiempo t va a usar la media y desviaci ́on de la
tasa en el tiempo t (una sola ventana). Adem ́as grafique la tasa transformada de
algunos ensayos de cada una de las condiciones como ejemplos.



(d) Obten y grafica el z-score (para cada neurona y clase) usando la tasa de cada
ventana particular y los ensayos correspondientes a cada clase para calcular el valor
medio y la desviaci ́on est ́andar con las que se computa la transformada z en cada
tiempo y clase. Para este inciso, adem ́as grafique la tasa transformada de algunos
ensayos de cada una de las clases como ejemplos.



(e) Discuta: ¿qu ́e diferencias observa entre los distintos c ́alculos realizados?


(f) Repita el primer inciso, pero ahora utilizando ventanas de 200ms con pasos de
40sms para las tres neuronas de S1 y ventanas 50 ms con pasos de 10 ms para las
dos neuronas de DPC. ¿Qu ́e diferencias observa?


"""





import pandas as pd
import seaborn as sns
import functions as fx
import numpy as np
import matplotlib.pyplot as plt



# Load the data from the CSV file
i_path = "/Users/juanpablomayaarteaga/Desktop/Practica2/"
o_path = fx.set_path(i_path + "Output/")
raster_path= fx.set_path(o_path + "Raster_plots/")
bin_path= fx.set_path(o_path + "Bin_plots/")
square_path= fx.set_path(o_path + "Square_plots/")
gaussian_path= fx.set_path(o_path + "Gaussian_plots/")
alpha_path= fx.set_path(o_path + "Alpha_plots/")
z_score_path= fx.set_path(o_path + "Z_Score_plots/")
sub_class_path=fx.set_path(o_path + "Sub_class_plots/")


"""
1. (1 a tiempo; 0.8 fuera de tiempo) Este ejercicio es sobre la transformada z (z-scores),
y vamos a usar los siguientes archivos: 
    
-de la tarea de discriminaci ́on de patrones TiemposNeuDPC1.csv, TiemposNeuDPC2.csv y TiemposNeuS1.csv; 

-de la tarea de detecci ́on DetNeuS1 B y NeuDetS1 D. 

Antes de empezar, calcule las tasas de disparo de las cinco neuronas: 
    
    -para las tres neuronas de S1 emplee ventanas de 50ms y pasos de 10 ms; 
    -para las dos de DPC, ventanas de 200ms y pasos de 40ms. 
    
Vamos a calcular la evoluci ́on temporal para diferentes z-scores promedio por clase y neurona. 
Para todos los incisos, grafique cada una de sus curvas con una barra de error que indique ± una
desviaci ́on est ́andar del correspondiente z-score.
"""



# Path of files
file_paths = [
    f'{i_path}TiemposNeuDPC1.csv',  
    f'{i_path}TiemposNeuDPC2.csv',
    f'{i_path}TiemposNeuS1.csv',
    f'{i_path}Neu_Det_S1B.csv',
    f'{i_path}Neu_Det_S1D.csv'
]




# Assign to each file a variable

#Tarea de Tiempos
data_DPC1 = fx.read_file_action_potentials(file_paths[0])
data_DPC2 = fx.read_file_action_potentials(file_paths[1])
data_S1 = fx.read_file_action_potentials(file_paths[2])
#Tarea de Deteccion
data_S1_DetB = fx.read_file_action_potentials(file_paths[3])
data_S1_DetD = fx.read_file_action_potentials(file_paths[4])



#Raster Plot: a spike train frequencies during each essay
fx.raster_plot(data_DPC1, path=raster_path, save_as="TiemposNeuDPC1")
fx.raster_plot(data_DPC2, path=raster_path, save_as="TiemposNeuDPC2")
fx.raster_plot(data_S1, path=raster_path, save_as="TiemposNeuS1")

fx.raster_plot(data_S1_DetB, path=raster_path, save_as="Neu_Det_S1B")
fx.raster_plot(data_S1_DetD, path=raster_path, save_as="Neu_Det_S1D")


DPC_window= 200
DPC_step=  40

#50 y 10 iniciales
S1_window= 50
S1_step= 10
 

#Binning Time
fx.binning_firing_rate(data=data_DPC1, bin_width_ms=DPC_window, path=bin_path, save_as='DPC1')
fx.binning_firing_rate(data=data_DPC2, bin_width_ms=DPC_window, path=bin_path, save_as='DPC"')
fx.binning_firing_rate(data=data_S1, bin_width_ms=S1_window, path=bin_path, save_as='S1')

fx.binning_firing_rate(data=data_S1_DetB, bin_width_ms=S1_window, path=bin_path, save_as='Neu_Det_S1B')
fx.binning_firing_rate(data=data_S1_DetD, bin_width_ms=S1_window, path=bin_path, save_as='Neu_Det_S1D')



#Sliding square window
fx.square_firing_rate(data=data_DPC1, window_size_ms=DPC_window, step_size_ms=DPC_step, path=square_path, save_as="DPC1")
fx.square_firing_rate(data=data_DPC2, window_size_ms=DPC_window, step_size_ms=DPC_step, path=square_path, save_as="DPC2")
fx.square_firing_rate(data=data_S1, window_size_ms=S1_window, step_size_ms=S1_step, path=square_path, save_as="S1")

fx.square_firing_rate(data=data_S1_DetB, window_size_ms=S1_window, step_size_ms=S1_step, path=square_path, save_as="Neu_Det_S1B")
fx.square_firing_rate(data=data_S1_DetD, window_size_ms=S1_window, step_size_ms=S1_step, path=square_path, save_as="Neu_Det_S1D")




#Sliding square window Z-score

fx.z_score_square_firing_rate(data=data_DPC1, window_size_ms=DPC_window, 
                              step_size_ms=DPC_step, line_color="red", 
                              z_score_color="gray", path=square_path, save_as="DPC_1")

fx.z_score_square_firing_rate(data=data_DPC2, window_size_ms=DPC_window, 
                              step_size_ms=DPC_step, path=square_path, save_as="DPC_2")

fx.z_score_square_firing_rate(data=data_S1, window_size_ms=S1_window, 
                              step_size_ms=S1_step, path=square_path, save_as="S1")



fx.z_score_square_firing_rate(data=data_S1_DetB, window_size_ms=S1_window, 
                              step_size_ms=S1_step, path=square_path, save_as="Neu_Det_S1_B")
fx.z_score_square_firing_rate(data=data_S1_DetD, window_size_ms=S1_window, 
                              step_size_ms=S1_step, path=square_path, save_as="Neu_Det_S1_D")





"""
(b) Obten y grafica el z-score (para cada neurona y clase) usando la tasa de las ventanas
comprendidas entre -2s y 0s (a este tipo de tasa le llamamos tasa basal) y todos los
ensayos (k = 60) para calcular el valor medio y la desviaci ́on est ́andar con las que
se computa la transformada z en cada clase.

"""






fx.z_score_basal_rate(data=data_DPC1, window_size_ms=DPC_window, 
                      step_size_ms=DPC_step, line_color="red", 
                      z_score_color="gray", path=square_path, save_as="DPC_1")

fx.z_score_square_firing_rate(data=data_DPC2, window_size_ms=DPC_window, 
                              step_size_ms=DPC_step, path=square_path, save_as="DPC_2")

fx.z_score_square_firing_rate(data=data_S1, window_size_ms=S1_window, 
                              step_size_ms=S1_step, path=square_path, save_as="S1")



fx.z_score_square_firing_rate(data=data_S1_DetB, window_size_ms=S1_window, 
                              step_size_ms=S1_step, path=square_path, save_as="Neu_Det_S1_B")
fx.z_score_square_firing_rate(data=data_S1_DetD, window_size_ms=S1_window, 
                              step_size_ms=S1_step, path=square_path, save_as="Neu_Det_S1_D")







fx.z_score_basal2task_firing_rate(data=data_DPC1, window_size_ms=DPC_window, 
                      step_size_ms=DPC_step, line_color="red", 
                      z_score_color="gray", path=square_path, save_as="DPC_1")


fx.z_score_basal2task_firing_rate(data=data_DPC2, window_size_ms=DPC_window, 
                              step_size_ms=DPC_step,line_color="red", 
                              z_score_color="gray", path=square_path, save_as="DPC_2")

fx.z_score_basal2task_firing_rate(data=data_S1, window_size_ms=S1_window, 
                              step_size_ms=S1_step,line_color="red", 
                              z_score_color="gray", path=square_path, save_as="S1")



fx.z_score_basal2task_firing_rate(data=data_S1_DetB, window_size_ms=S1_window, 
                              step_size_ms=S1_step, line_color="red", 
                              z_score_color="gray",path=square_path, save_as="Neu_Det_S1_B")

fx.z_score_basal2task_firing_rate(data=data_S1_DetD, window_size_ms=S1_window, 
                              step_size_ms=S1_step, line_color="red", 
                              z_score_color="gray", path=square_path, save_as="Neu_Det_S1_D")






fx.z_firing_raster_plot(data=data_DPC1,
                      window_size_ms=S1_window, step_size_ms=S1_step, 
                      path=square_path, save_as="data_DPC1")












fx.threshold_firing_rate(data=data_DPC1,
                      window_size_ms=S1_window, step_size_ms=S1_step, 
                      path=z_score_path, save_as="DPC1", threshold=1.8)

fx.threshold_firing_rate(data=data_DPC2,
                      window_size_ms=S1_window, step_size_ms=S1_step, 
                      path=z_score_path, save_as="DPC2", threshold=1.3)

fx.threshold_firing_rate(data=data_S1,
                      window_size_ms=S1_window, step_size_ms=S1_step, 
                      path=z_score_path, save_as="S1", threshold=1.8)

fx.threshold_firing_rate(data=data_S1_DetB,
                      window_size_ms=S1_window, step_size_ms=S1_step, 
                      path=z_score_path, save_as="S1_DetB", threshold=1.4)

fx.threshold_firing_rate(data=data_S1_DetD,
                      window_size_ms=S1_window, step_size_ms=S1_step, 
                      path=z_score_path, save_as="S1_DetB", threshold=1.2)




"""
(c) Obten y grafica el z-score (para cada neurona y clase) usando la tasa de cada ventana
de tiempo particular y todos los ensayos para calcular el valor medio y la desviaci ́on
est ́andar con las que se computa la transformada z en cada tiempo y clase. Esto
es, para computar el z-score en el tiempo t va a usar la media y desviaci ́on de la
tasa en el tiempo t (una sola ventana). Adem ́as grafique la tasa transformada de
algunos ensayos de cada una de las condiciones como ejemplos.



"""




fx.square_firing_rate_by_class(data=data_S1_DetD,
                         window_size_ms=S1_window, step_size_ms=S1_step, 
                         path=sub_class_path, save_as="S1_DetB")







fx.z_score_firing_rate_by_class_subplots(data=data_S1_DetB,
                         window_size_ms=S1_window, step_size_ms=S1_step, 
                         path=sub_class_path, save_as="S1_DetB")

fx.z_score_firing_rate_by_class_subplots(data=data_S1_DetD,
                         window_size_ms=S1_window, step_size_ms=S1_step, 
                         path=sub_class_path, save_as="S1_DetD")








fx.z_score_firing_rate_by_class_single_plot(data=data_S1_DetD,
                         window_size_ms=S1_window, step_size_ms=S1_step, 
                         path=sub_class_path, save_as="S1_DetB")







"""



(d) Obten y grafica el z-score (para cada neurona y clase) usando la tasa de cada
ventana particular y los ensayos correspondientes a cada clase para calcular el valor
medio y la desviaci ́on est ́andar con las que se computa la transformada z en cada
tiempo y clase. Para este inciso, adem ́as grafique la tasa transformada de algunos
ensayos de cada una de las clases como ejemplos.



(e) Discuta: ¿qu ́e diferencias observa entre los distintos c ́alculos realizados?


(f) Repita el primer inciso, pero ahora utilizando ventanas de 200ms con pasos de
40sms para las tres neuronas de S1 y ventanas 50 ms con pasos de 10 ms para las
dos neuronas de DPC. ¿Qu ́e diferencias observa?


"""










