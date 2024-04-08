#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:02:46 2023

@author: juanpablomayaarteaga
"""

import os
import pandas as pd
import seaborn as sns
import functions as fx
import numpy as np
import matplotlib.pyplot as plt


def set_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
    return path


def read_file_action_potentials(file_path):
    """
    This code parses neuron action potential timing data from a CSV file, 
    structured with trials and blocks of data separated by empty lines. 
    
    The original csv file contains 63 lines in blocks of 15 and then an empty line that divides the blocks
    and 300 columns not all the columns have values, there are some empty cells.
    
    So there are 4 blocks divided by 3 empty lines
    
    Each line represents an essay. So this code returns a list with 4 list that contain each line in a list (nested list)
    
    
    Parameters:
    - file_path: str, path to the CSV file containing the AP data.
    
    -block: is a list. Each block is a list of trials
    -current_block: temporarily hold the trials for the current block being processed.
    Current block read line by line, and add to block list each block 
        
    Returns:
    - A list of blocks, where each block is a list of trials, and each trial is a list of time points.
   
    """
    blocks = []
    current_block = []
    with open(file_path, 'r') as file: #"r" read mode
        for line in file: 
            # Check if the line is empty (a block separator indicated in the practice. Other dataset can be stored in different way)
            if line.strip() == '':
                if current_block:  # If the current block is not empty, add it to the blocks list
                    blocks.append(current_block)
                    current_block = []  # Reset the current block
            else:
                # Convert the line to a list of floats (times)
                trial_data = list(map(float, line.strip().split(','))) #Processes a non-empty line by commas and convert that data into floats
                current_block.append(trial_data)
        if current_block:  # Add the last block if not empty
            blocks.append(current_block)
    return blocks








def raster_plot(blocks, path, save_as):
    # Determine the total number of trials across all blocks
    num_trials = sum(len(block) for block in blocks)
    trial_counter = 0
    plt.figure(figsize=(10, 8))

    # Plot each block
    for block in blocks:
        for trial in block:
            # Each trial's spike times are plotted at an incrementing y-value (trial number)
            plt.eventplot(trial, lineoffsets=trial_counter, linelengths=0.8)
            trial_counter += 1

    plt.xlabel('Time (ms)', fontsize=12, fontweight='bold')
    plt.ylabel('Trial', fontsize=12, fontweight='bold')
    plt.title(f'Raster Plot:  {save_as.replace("_", " ")}', fontsize=14, fontweight='bold')
    
   
    # Define the file name based on parameters
    file_name = f"Raster_plot_{save_as.replace(' ', '_')}.png"
    plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    full_path = f"{path}{file_name}"  # Construct the full file path


    print(f"Plot saved to {full_path}")  # Optional: Print confirmation message







import numpy as np
import matplotlib.pyplot as plt

def binning_firing_rate(data, bin_width_ms, path, save_as=''):
    """
    Calculates and plots the firing rate of neuronal data over time,
    using a discrete time binning approach.
    
    Parameters:
    - data: Data to calculate firing rates from. This should be a list of lists,
            where each inner list contains spike times for a single trial.
    - bin_width_ms: Width of each time bin in milliseconds.
    - path: Output path to save the plot.
    - save_as: Additional text to append to the file name when storing the plot.
    """
    
    # Concatenate each nested list into a single array and then, concatenate each block
    spike_times = np.concatenate([np.concatenate(block) for block in data])
    
    # Define the observation period based on the earliest and latest spike times
    start_time = min(spike_times)
    end_time = max(spike_times)
    
    # Convert the bin width from milliseconds to seconds
    bin_width = bin_width_ms / 1000
    
    # Create bins for the entire observation period
    bins = np.arange(start_time, end_time + bin_width, bin_width)
    
    # Count the number of spikes in each bin
    spike_counts, _ = np.histogram(spike_times, bins)
    
    # Calculate the firing rate for each bin (spikes per second)
    firing_rates = spike_counts / bin_width
    
    # Prepare the time axis for plotting (middle of each bin)
    times = bins[:-1] + bin_width / 2
    
    # Plot the firing rate
    plt.figure(figsize=(12, 7))
    plt.bar(times, firing_rates, width=bin_width, align='center')
    plt.title(f'{save_as.replace("_", " ")} Binned Firing Rate: Bin Width {bin_width_ms}ms', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
    
    # Save the plot
    file_name = f"binned_firing_rate_bw_{bin_width_ms}_{save_as.replace(' ', '_')}.png"
    plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()






def square_firing_rate(data, window_size_ms, step_size_ms, path, save_as=''):
    """
    This code calculate and plot the firing rate of neuronal data over time, 
    using a sliding square causal window approach.

    The window size determines the temporal resolution of your firing rate analysis. 
    A larger window size will smooth out the firing rate calculation over a longer period, 
    potentially missing brief spikes in activity. 
    In contrast, a smaller window size provides a more granular view of the firing rate fluctuations 
    but may be more susceptible to noise or variance in the spike counts.
    
    The step size determines how much overlap there is between consecutive windows. 
    A step size smaller than the window size means that consecutive windows will overlap,
    providing a smoother and potentially more correlated series of firing rate measurements. 
    A step size equal to the window size means no overlap (each spike is only counted once), 
    leading to more independent measurements but possibly a choppier firing rate profile.
    
    To visualize these concepts, imagine your data as a timeline of spikes recorded from a neuron. 
    The window size is like a sliding measuring tape you place along this timeline to 
    count spikes within that interval. The step size is how far you move this measuring tape 
    each time you take a new measurement. If your window size is 100 ms and your step size is 50 ms,
    it means you measure the firing rate over 100 ms intervals, shifting the window by 50 ms 
    after each measurement. This approach gives you a detailed yet overlapping picture 
    of how the neuron's firing rate changes over time.

    Parameters:
    - data: Data to calculate firing rates from.
    - window_size_ms: Size of the window in milliseconds.
    - step_size_ms: Step size in milliseconds.
    - path: Output path to save the plot.
    - save_as: Additional text to append to the plot  when storing it.
    
    """
    
    #Concatenate each nested list into a single array and then, concatenate each block
    spike_times = np.concatenate([np.concatenate(block) for block in data])
    
    #This defines the observation period: Set the start and end of the observation period based on the earliest and latest spike times.
    start_time = min(spike_times)
    end_time = max(spike_times)
    
    # The window and step sizes are converted from milliseconds to seconds to match the time unit of spike_times
    window_size = window_size_ms / 1000
    step_size = step_size_ms / 1000
    
    # Initialize the time and firing rate arrays
    times = np.arange(start_time, end_time, step_size) #creates an array of times starting from start_time to end_time with increments of step_size 
    firing_rates = [] #initializes an empty list to store the calculated firing rates.
    
    # Calculate firing rate for each window.
    # For each window, it counts the number of spikes (spikes_in_window) and calculates the firing rate as the number of spikes divided by the window size in seconds, appending each rate to firing_rates.
    for time in times:
        window_start = time
        window_end = time + window_size #Defining a Window: This window moves forward in time by step_size seconds with each iteration.
        spikes_in_window = ((spike_times >= window_start) & (spike_times < window_end)).sum() #counts the number of spikes occurring within the current window. It does this by creating a boolean array where each element is True if the corresponding spike time is between window_start and window_end
        firing_rate = (spikes_in_window / window_size)
        firing_rates.append(firing_rate)
    
    # Plot the firing rate
    plt.figure(figsize=(12, 7))
    plt.plot(times + window_size / 2, firing_rates, linewidth=2)  # Plot the middle of the window
    plt.title(f'{save_as.replace("_", " ")} Firing Rate: Window Size {window_size_ms}ms, Step Size {step_size_ms}ms', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
    
    # Define the file name based on parameters
    file_name = f"square_firing_rate_ws_{window_size_ms}_ss_{step_size_ms}_{save_as.replace(' ', '_')}.png"
    plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()






from scipy.signal import gaussian
from scipy.ndimage import convolve1d

def gaussian_firing_rate(data, window_size_ms, step_size_ms, sigma, path, save_as=''):
    """
    Calculates and plots the firing rate of neuronal data over time, 
    using a sliding Gaussian window approach.
    
    Parameters:
    - data: Data to calculate firing rates from.
    - window_size_ms: Size of the window in milliseconds.
    - step_size_ms: Step size in milliseconds.
    - sigma: Standard deviation of the Gaussian window, controlling the smoothing.
    - path: Output path to save the plot.
    - save_as: Additional text to append to the plot when storing it.
    """
    
    # Concatenate each nested list into a single array and then, concatenate each block
    spike_times = np.concatenate([np.concatenate(block) for block in data])
    
    # Define the observation period based on the earliest and latest spike times
    start_time = min(spike_times)
    end_time = max(spike_times)
    
    # Convert window and step sizes from milliseconds to seconds
    window_size = window_size_ms / 1000
    step_size = step_size_ms / 1000
    
    # Calculate the number of points in the Gaussian window based on the step size
    num_points = int(window_size / step_size) + 1
    
    # Generate a Gaussian window using the provided sigma value
    gaussian_window = gaussian(num_points, std=sigma)
    gaussian_window /= np.sum(gaussian_window)  # Normalize the window to sum to 1
    
    # Create a spike histogram with a bin for each step
    bins = np.arange(start_time, end_time + step_size, step_size)
    spike_counts, _ = np.histogram(spike_times, bins)
    
    # Use convolution to apply the Gaussian window to the spike histogram
    firing_rates = convolve1d(spike_counts, gaussian_window, mode='constant', cval=0.0) / step_size
    
    # Adjust the length of firing_rates if necessary
    firing_rates = firing_rates[:len(bins)-1]
    
    # Prepare the time axis for plotting
    times = bins[:-1] + step_size / 2
    
    # Plot the firing rate
    plt.figure(figsize=(12, 7))
    plt.plot(times, firing_rates, linewidth=2)
    plt.title(f'{save_as.replace("_", " ")} Gaussian Firing Rate: Window Size {window_size_ms}ms, Step Size {step_size_ms}ms, Sigma {sigma}', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
    
    # Save the plot
    file_name = f"gaussian_firing_rate_ws_{window_size_ms}_ss_{step_size_ms}_sigma_{sigma}_{save_as}.png"
    plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()






def alpha_function_window(length, alpha):
    """
    Generates an alpha function window.

    Parameters:
    - length: Number of points in the window.
    - alpha: Parameter controlling the shape of the alpha function.

    Returns:
    - A numpy array containing the alpha function window.
    """
    t = np.arange(0, length)
    window = t * np.exp(-alpha * t)
    return window / np.sum(window)  # Normalize the window to sum to 1

def alpha_firing_rate(data, window_size_ms, step_size_ms, alpha, path, save_as=''):
    """
    Calculates and plots the firing rate of neuronal data over time, 
    using a sliding alpha function window approach.
    
    Parameters:
    - data: Data to calculate firing rates from.
    - window_size_ms: Size of the window in milliseconds.
    - step_size_ms: Step size in milliseconds.
    - alpha: Alpha parameter for the alpha function window.
    - path: Output path to save the plot.
    - save_as: Additional text to append to the plot when storing it.
    """
    
    # Concatenate each nested list into a single array and then, concatenate each block
    spike_times = np.concatenate([np.concatenate(block) for block in data])
    
    # Define the observation period based on the earliest and latest spike times
    start_time = min(spike_times)
    end_time = max(spike_times)
    
    # Convert window and step sizes from milliseconds to seconds
    window_size = window_size_ms / 1000
    step_size = step_size_ms / 1000
    
    # Calculate the number of points in the alpha function window based on the step size
    num_points = int(window_size / step_size) + 1
    
    # Generate an alpha function window
    alpha_window = alpha_function_window(num_points, alpha)
    
    # Create a spike histogram with a bin for each step
    bins = np.arange(start_time, end_time + step_size, step_size)
    spike_counts, _ = np.histogram(spike_times, bins)
    
    # Use convolution to apply the alpha function window to the spike histogram
    firing_rates = convolve1d(spike_counts, alpha_window, mode='constant', cval=0.0) / step_size
    
    # Adjust the length of firing_rates if necessary
    firing_rates = firing_rates[:len(bins)-1]
    
    # Prepare the time axis for plotting
    times = bins[:-1] + step_size / 2
    
    # Plot the firing rate
    plt.figure(figsize=(12, 7))
    plt.plot(times, firing_rates, linewidth=2)
    plt.title(f'{save_as.replace("_", " ")} Alpha Firing Rate: Window Size {window_size_ms}ms, Step Size {step_size_ms}ms, Alpha {alpha}', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Firing Rate (Hz)', fontsize=12, fontweight='bold')
    
    # Save the plot
    file_name = f"alpha_firing_rate_ws_{window_size_ms}_ss_{step_size_ms}_alpha_{alpha}_{save_as}.png"
    plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()







def z_score_square_firing_rate(data, window_size_ms, step_size_ms, path, save_as='', line_color='blue', z_score_color='gray'):
    """
    Calculates, plots, and saves the z-score of the firing rate of neuronal data over time,
    using a sliding square causal window approach. It visualizes the mean z-score and its 
    standard deviation as error bars at each time point.
    
    Parameters:
    - data: Data to calculate firing rates from.
    - window_size_ms: Size of the window in milliseconds.
    - step_size_ms: Step size in milliseconds.
    - path: Output path to save the plot.
    - save_as: Additional text to append to the plot file name.
    """

        
    # Convert window and step sizes from ms to seconds
    window_size = window_size_ms / 1000
    step_size = step_size_ms / 1000
    
    # Concatenate spike times from all trials into a single array
    spike_times = np.concatenate([np.concatenate(trial) for trial in data])
    
    # Define the observation period
    start_time = np.min(spike_times)
    end_time = np.max(spike_times)
    
    # Initialize arrays for time and firing rates
    times = np.arange(start_time, end_time - window_size, step_size)
    firing_rates = np.zeros_like(times, dtype=float)
    
    # Calculate firing rate for each window
    for i, time in enumerate(times):
        window_start = time
        window_end = time + window_size
        spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
        firing_rate = spikes_in_window / window_size
        firing_rates[i] = firing_rate
    
    # Calculate the z-score for the firing rates
    mean_firing_rate = np.mean(firing_rates)
    std_firing_rate = np.std(firing_rates)
    z_scores = (firing_rates - mean_firing_rate) / std_firing_rate
    
    # Calculate standard deviation of z-scores for error bars
    std_dev_z_scores = np.std(z_scores)
    
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(times + window_size / 2, z_scores, '-',  color=line_color, linewidth=4, label='Z-Score')
    
    """
    Line Styles
        - or solid: Solid line
        -- or dashed: Dashed line
        -. or dashdot: Dash-dot line
        : or dotted: Dotted line
        None, or '': No line (only markers)
        
    Marker Styles
        .: Point marker
        o: Circle marker
        s: Square marker
        d: Diamond marker
        ^: Triangle up marker
        <: Triangle left marker
        >: Triangle right marker
        v: Triangle down marker
        +: Plus marker
        x: X marker
        *: Star marker
        |: Vertical line marker
        _: Horizontal line marker
    """
    
    # Use fill_between to shade the area for standard deviation in light red
    plt.fill_between(times + window_size / 2, 
                     z_scores - std_dev_z_scores, 
                     z_scores + std_dev_z_scores, 
                     color=z_score_color, alpha=0.4, 
                     label='±1 SD')
    
    plt.title(f'Z-Score of Firing Rate Over Time ({save_as.replace("_", " ")})', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Z-Score of Firing Rate', fontsize=12, fontweight='bold')
    plt.legend()
    
    # Save the plot
    file_name = f"z_score_firing_rate_{save_as.replace(' ', '_')}.png"
    plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()



def z_score_basal_rate(data, window_size_ms, step_size_ms, line_color='blue', z_score_color='gray', path='/', save_as=''):
    """
    Calculates, plots, and saves the z-score of the basal firing rate (before time 0) of neuronal data over time,
    using a sliding square causal window approach. It visualizes the mean z-score and its standard deviation
    as error bars at each time point.

    Parameters:
    - data: Data to calculate firing rates from.
    - window_size_ms: Size of the window in milliseconds.
    - step_size_ms: Step size in milliseconds.
    - line_color: Color of the mean line plot.
    - z_score_color: Color of the standard deviation shading.
    - path: Output path to save the plot.
    - save_as: Additional text to append to the plot file name.
    """
    # Convert window and step sizes from ms to seconds
    window_size = window_size_ms / 1000
    step_size = step_size_ms / 1000

    # Concatenate spike times from all trials into a single array
    spike_times = np.concatenate([np.concatenate(trial) for trial in data])

    # Focus on the basal period (before time 0)
    spike_times = spike_times[spike_times < 0]

    # Define the observation period
    start_time = np.min(spike_times)
    end_time = 0  # Basal period ends at time 0

    # Initialize arrays for time and firing rates
    times = np.arange(start_time, end_time, step_size)
    firing_rates = np.zeros(len(times))

    # Calculate firing rate for each window
    for i, time in enumerate(times):
        window_start = time
        window_end = time + window_size
        spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
        firing_rate = spikes_in_window / window_size
        firing_rates[i] = firing_rate

    # Calculate the z-score for the basal firing rates
    mean_firing_rate = np.mean(firing_rates)
    std_firing_rate = np.std(firing_rates)
    z_scores = (firing_rates - mean_firing_rate) / std_firing_rate

    # Calculate standard deviation of z-scores for error bars
    std_dev_z_scores = np.std(z_scores)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(times + window_size / 2, z_scores, '-', color=line_color, label='Z-Score')

    # Use fill_between to shade the area for standard deviation
    plt.fill_between(times + window_size / 2, z_scores - std_dev_z_scores, z_scores + std_dev_z_scores, color=z_score_color, alpha=0.4, label='±1 SD')

    plt.title(f'Basal Z-Score of Firing Rate: {save_as}', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Z-Score of Firing Rate', fontsize=12, fontweight='bold')
    plt.legend()

    # Save the plot
    file_name = f"z_score_basal_rate_{save_as.replace(' ', '_')}.png"
    plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

# Example usage (assuming data preparation and path specification have been done)
# z_score_basal_rate(data_DPC1, 100, 50, path='/path/to/save', save_as='example_basal_rate')


def z_score_task_rate(data, window_size_ms, step_size_ms, line_color='blue', z_score_color='gray', path='/mnt/data', save_as=''):
    """
    Calculates, plots, and saves the z-score of the task-related firing rate of neuronal data over time,
    focusing on the period from time 0 to the maximum spike time. It visualizes the mean z-score and its 
    standard deviation as error bars at each time point, highlighting neuronal activity during the task.

    Parameters:
    - data: Data to calculate firing rates from, focusing on the period from 0 onwards.
    - window_size_ms: Size of the window in milliseconds.
    - step_size_ms: Step size in milliseconds.
    - line_color: Color of the mean line in the plot.
    - z_score_color: Color of the shaded standard deviation area in the plot.
    - path: Output path to save the plot.
    - save_as: Additional text to append to the plot file name.
    """

    
    # Convert window and step sizes from ms to seconds
    window_size = window_size_ms / 1000
    step_size = step_size_ms / 1000
    
    # Concatenate spike times from all trials into a single array, focusing on spikes from time 0 onwards
    spike_times = np.concatenate([np.concatenate(trial) for trial in data if len(trial) > 0])
    task_spike_times = spike_times[spike_times >= 0]
    
    # Define the task period
    start_time = 0
    end_time = np.max(task_spike_times)
    
    # Initialize arrays for time and firing rates
    times = np.arange(start_time, end_time, step_size)
    firing_rates = np.zeros_like(times, dtype=float)
    
    # Calculate firing rate for each window
    for i, time in enumerate(times):
        window_start = time
        window_end = time + window_size
        spikes_in_window = np.sum((task_spike_times >= window_start) & (task_spike_times < window_end))
        firing_rate = spikes_in_window / window_size
        firing_rates[i] = firing_rate
    
    # Calculate the z-score for the firing rates
    mean_firing_rate = np.mean(firing_rates)
    std_firing_rate = np.std(firing_rates)
    z_scores = (firing_rates - mean_firing_rate) / std_firing_rate
    
    # Calculate standard deviation of z-scores for error bars
    std_dev_z_scores = np.std(z_scores)
    
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(times + window_size / 2, z_scores, '-', color=line_color, linewidth=2, label='Z-Score')
    
    plt.fill_between(times + window_size / 2, z_scores - std_dev_z_scores, z_scores + std_dev_z_scores, color=z_score_color, alpha=0.4, label='±1 SD')
    
    plt.title(f'Z-Score of Task-Related Firing Rate: {save_as}', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Z-Score of Firing Rate', fontsize=12, fontweight='bold')
    plt.legend()
    
    # Save the plot
    file_name = f"z_score_task_rate_{save_as.replace(' ', '_')}.png"
    plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    return f"{path}/{file_name}"

# Example usage of the function would require actual data, window size, step size, and optional parameters for colors and saving.
# Please provide specific parameters and data to run this function with your dataset.






def z_score_basal2task_firing_rate(data, window_size_ms, step_size_ms, path, save_as='', line_color='blue', z_score_color='gray'):
    """
    Calculates, plots, and saves the z-score of the firing rate of neuronal data over time,
    using a sliding square causal window approach. It visualizes the mean z-score and its 
    standard deviation as error bars at each time point.
    
    Parameters:
    - data: Data to calculate firing rates from.
    - window_size_ms: Size of the window in milliseconds.
    - step_size_ms: Step size in milliseconds.
    - path: Output path to save the plot.
    - save_as: Additional text to append to the plot file name.
    """

        
    # Convert window and step sizes from ms to seconds
    window_size = window_size_ms / 1000
    step_size = step_size_ms / 1000
    
    # Concatenate spike times from all trials into a single array
    spike_times = np.concatenate([np.concatenate(trial) for trial in data])
    
    # Define the observation period
    start_time = np.min(spike_times)
    end_time = np.max(spike_times)
    
    # Initialize arrays for time and firing rates
    times = np.arange(start_time, end_time - window_size, step_size)
    firing_rates = np.zeros_like(times, dtype=float)
    
    # Calculate firing rate for each window
    for i, time in enumerate(times):
        window_start = time
        window_end = time + window_size
        spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
        firing_rate = spikes_in_window / window_size
        firing_rates[i] = firing_rate
    
    # Calculate the z-score for the firing rates
    mean_firing_rate = np.mean(firing_rates)
    std_firing_rate = np.std(firing_rates)
    z_scores = (firing_rates - mean_firing_rate) / std_firing_rate
    
    # Calculate standard deviation of z-scores for error bars
    std_dev_z_scores = np.std(z_scores)
    
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(times + window_size / 2, 
             z_scores, '-',  
             color=line_color, 
             linewidth=4, 
             label='Z-Score')
    
    # Use fill_between to shade the area for standard deviation
    plt.fill_between(times + window_size / 2, 
                     z_scores - std_dev_z_scores, 
                     z_scores + std_dev_z_scores, 
                     color=z_score_color, alpha=0.4, 
                     label='±1 SD')
    
    # Drawing the vertical line at time 0 and background colors
    plt.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.5)
    plt.axvspan(min(times), 0, color='gray', alpha=0.2)
    plt.axvspan(0, max(times) + window_size, color='lightblue', alpha=0.4)
    
    plt.title(f'Z-Score of Firing Rate: {save_as.replace("_", " ")}', 
              fontsize=14, fontweight='bold')
    
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Z-Score of Firing Rate', fontsize=12, fontweight='bold')
    plt.legend()
    
    # Save the plot
    file_name = f"z_score_firing_rate_{save_as.replace(' ', '_')}.png"
    plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    



def firing_raster_plot(data, window_size_ms, step_size_ms, path, save_as=''):
    window_size = window_size_ms / 1000  # Convert milliseconds to seconds
    step_size = step_size_ms / 1000  # Convert milliseconds to seconds
    spike_times = np.concatenate([np.concatenate(trial) for trial in data])
    start_time = np.min(spike_times)
    end_time = np.max(spike_times)
    times = np.arange(start_time, end_time - window_size, step_size)
    firing_rates = np.zeros_like(times, dtype=float)
    for i, time in enumerate(times):
        window_start = time
        window_end = time + window_size
        spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
        firing_rate = spikes_in_window / window_size
        firing_rates[i] = firing_rate

    # Create figure with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [1, 3]})

    # Firing rate plot adjustments
    axs[0].plot(times + window_size / 2, firing_rates, '-', color='red', linewidth=4, label='Firing Rate')
    axs[0].axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.2)
    axs[0].axvspan(start_time, 0, color='gray', alpha=0.2)
    axs[0].axvspan(0, end_time, color='blue', alpha=0.2)
    axs[0].set_ylabel('Firing Rate (spikes/s)', fontsize=12, fontweight='bold')
    #axs[0].legend()
    axs[0].set_xticklabels([])  # Remove x-axis labels from firing rate plot

    # Raster plot adjustments
    trial_counter = 0
    for block in data:
        for trial in block:
            axs[1].eventplot(trial, lineoffsets=trial_counter, linelengths=0.8, color="red")
            trial_counter += 1
    axs[1].axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.2)
    axs[1].axvspan(start_time, 0, color='gray', alpha=0.2)
    axs[1].axvspan(0, end_time, color='blue', alpha=0.2)
    axs[1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Trial', fontsize=12, fontweight='bold')

    # Adjust the layout and save the combined plot
    plt.tight_layout()
    file_name = f"Raster_and_Firing_plot_{save_as.replace(' ', '_')}.png"
    plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Combined plot saved to {path}/{file_name}")



def z_firing_raster_plot(data, window_size_ms, step_size_ms, path, save_as=''):
    # Preliminary calculations for the z-score plot
    window_size = window_size_ms / 1000
    step_size = step_size_ms / 1000
    spike_times = np.concatenate([np.concatenate(trial) for trial in data])
    start_time = np.min(spike_times)
    end_time = np.max(spike_times)
    times = np.arange(start_time, end_time - window_size, step_size)
    firing_rates = np.zeros_like(times, dtype=float)
    for i, time in enumerate(times):
        window_start = time
        window_end = time + window_size
        spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
        firing_rate = spikes_in_window / window_size
        firing_rates[i] = firing_rate
    mean_firing_rate = np.mean(firing_rates)
    std_firing_rate = np.std(firing_rates)
    z_scores = (firing_rates - mean_firing_rate) / std_firing_rate
    std_dev_z_scores = np.std(z_scores)

    # Create figure with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [1, 3]})

    # Z-score plot adjustments
    axs[0].plot(times + window_size / 2, z_scores, '-', color='red', linewidth=4, label='Z-Score')
    axs[0].fill_between(times + window_size / 2, z_scores - std_dev_z_scores, z_scores + std_dev_z_scores, color='gray', alpha=0.4, label='±1 SD')
    axs[0].axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.2)
    axs[0].axvspan(start_time, 0, color='gray', alpha=0.2)
    axs[0].axvspan(0, end_time, color='blue', alpha=0.2)
    axs[0].set_ylabel('Z-Score of Firing Rate', fontsize=12, fontweight='bold')
    axs[0].legend()
    axs[0].set_xticklabels([])  # Remove x-axis labels from z-score plot

    # Raster plot adjustments
    trial_counter = 0
    for block in data:
        for trial in block:
            axs[1].eventplot(trial, lineoffsets=trial_counter, linelengths=0.8, color="red")
            trial_counter += 1
    axs[1].axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.2)
    axs[1].axvspan(start_time, 0, color='gray', alpha=0.2)
    axs[1].axvspan(0, end_time, color='blue', alpha=0.2)
    axs[1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Trial', fontsize=12, fontweight='bold')

    # Adjust the layout and save the combined plot
    plt.tight_layout()
    file_name = f"Raster_and_Firing_plot_{save_as.replace(' ', '_')}.png"
    plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Combined plot saved to {path}/{file_name}")








def threshold_firing_rate(data, window_size_ms, step_size_ms, path, save_as='', threshold=2):
    # Preliminary calculations for the z-score plot
    window_size = window_size_ms / 1000
    step_size = step_size_ms / 1000
    spike_times = np.concatenate([np.concatenate(trial) for trial in data])
    start_time = np.min(spike_times)
    end_time = np.max(spike_times)
    times = np.arange(start_time, end_time - window_size, step_size)
    firing_rates = np.zeros_like(times, dtype=float)
    
    for i, time in enumerate(times):
        window_start = time
        window_end = time + window_size
        spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
        firing_rate = spikes_in_window / window_size
        firing_rates[i] = firing_rate
    
    mean_firing_rate = np.mean(firing_rates)
    std_firing_rate = np.std(firing_rates)
    z_scores = (firing_rates - mean_firing_rate) / std_firing_rate
    std_dev_z_scores = np.std(z_scores)

    # Identify regions where z-score exceeds the threshold
    threshold_exceeded = z_scores > threshold

    # Create figure with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [1, 3]})

    # Z-Score Plot
    axs[0].plot(times + window_size / 2, z_scores, '-', color='royalblue', linewidth=4, label='Z-Score')
    axs[0].fill_between(times + window_size / 2, z_scores - std_dev_z_scores, z_scores + std_dev_z_scores, color='gray', alpha=0.4, label='±1 SD')
    axs[0].set_ylabel('Z-Score of Firing Rate', fontsize=12, fontweight='bold')
    axs[0].legend()
    axs[0].set_xticklabels([])  # Remove x-axis labels from z-score plot
    axs[0].set_title(f'{save_as.replace("_", " ")} - threshold: {threshold}', fontsize=14, fontweight='bold')

    # Raster Plot
    trial_counter = 0
    for block in data:
        for trial in block:
            axs[1].eventplot(trial, lineoffsets=trial_counter, linelengths=0.8, color="royalblue")
            trial_counter += 1
    axs[1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Trial', fontsize=12, fontweight='bold')

    # Shade background before time 0 in gray
    for ax in axs:
        ax.axvspan(start_time, 0, color='gray', alpha=0.2)

    # Shade areas exceeding the threshold in blue
    for ax in axs:
        for start, end in zip(times[threshold_exceeded], times[threshold_exceeded] + step_size):
            ax.axvspan(start, end, color='red', alpha=0.2)

    # Adjust layout and save
    plt.tight_layout()
    file_name = f"Plot_{save_as.replace(' ', '_')}_threshold_{threshold}.png"
    plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Combined plot saved to {path}/{file_name}")

           




def square_firing_rate_by_class(data, window_size_ms, step_size_ms, path, save_as=''):
    """
    Calculate and plot the firing rate of neuronal data separated into six classes, 
    each represented by a list within the data. Each list contains nested lists of spike times.
    The function processes each class, calculates the mean firing rate over a sliding square window, 
    and plots the results in a subplot grid of 3x2 with each class represented by a different color line.

    Parameters:
    - data: List of six lists, each containing nested lists of spike times.
    - window_size_ms: Size of the sliding window in milliseconds.
    - step_size_ms: Step size for the sliding window in milliseconds.
    - path: Output path to save the plot.
    - save_as: Filename suffix for the saved plot.
    """
    # Define a list of colors for each class
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange']
    fig, axs = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f'Firing Rate by Class: Window Size {window_size_ms}ms, Step Size {step_size_ms}ms', fontsize=16, fontweight='bold')

    for i, class_data in enumerate(data):
        # Calculate the mean firing rate for the current class
        spike_times = np.concatenate(class_data)
        start_time = min(spike_times)
        end_time = max(spike_times)
        window_size = window_size_ms / 1000
        step_size = step_size_ms / 1000
        times = np.arange(start_time, end_time - window_size, step_size)
        firing_rates = []

        for time in times:
            window_start = time
            window_end = time + window_size
            spikes_in_window = ((spike_times >= window_start) & (spike_times < window_end)).sum()
            firing_rate = spikes_in_window / window_size
            firing_rates.append(firing_rate)
        
        # Plotting with specified color
        ax = axs.flatten()[i]
        ax.plot(times + window_size / 2, firing_rates, color=colors[i], linewidth=2)
        ax.set_title(f'Class {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')

    # Save the plot
    if save_as:
        plt.savefig(f"{path}/{save_as}_firing_rate_by_class.png", dpi=300)

    plt.show()



def subplots_firing_rate_from_dictionary(flattened_data, window_size_ms, step_size_ms, path, title="", save_as='', raws=4, columns=4):
    """
    Plot the firing rate for each dataset in the flattened_data dictionary.
    Each dataset's firing rate is calculated and plotted in a subplot grid of 3x4.

    Parameters:
    - flattened_data: Dictionary with keys as dataset names and values as flattened spike times arrays.
    - window_size_ms: Size of the sliding window in milliseconds.
    - step_size_ms: Step size for the sliding window in milliseconds.
    - path: Output path to save the plot.
    - save_as: Filename suffix for the saved plot.
    """
    # Setting up the subplot grid
    fig, axs = plt.subplots(raws, columns, figsize=(15, 20), constrained_layout=True)
    fig.suptitle(f'{title} Firing Rate: Window Size {window_size_ms}ms, Step Size {step_size_ms}ms', fontsize=16, fontweight='bold')

    # Calculate and plot firing rate for each dataset
    for i, (dataset_name, spike_times) in enumerate(flattened_data.items()):
        if i >= 12:  # To ensure it doesn't try to plot more than the grid size
            break
        
        start_time = np.min(spike_times)
        end_time = np.max(spike_times)
        window_size = window_size_ms / 1000  # Convert to seconds
        step_size = step_size_ms / 1000  # Convert to seconds
        times = np.arange(start_time, end_time - window_size, step_size)
        firing_rates = []

        for time in times:
            window_start = time
            window_end = time + window_size
            spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
            firing_rate = spikes_in_window / window_size
            firing_rates.append(firing_rate)
        
        # Plotting
        ax = axs.flatten()[i]
        ax.plot(times + window_size / 2, firing_rates, linewidth=2)
        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=10)

    # Adjust subplots that are not used (if any)
    for j in range(i + 1, raws*columns):
        axs.flatten()[j].axis('off')

    # Save the plot
    if save_as:
        file_name = f"Subplots_{title}_firing_rate.png"
        plt.savefig(f"{path}/{file_name}", dpi=300)

    plt.show()
    print(f"Plot saved to {path}{file_name}")
    
    


def subplots_z_firing_rate_from_dictionary(flattened_data, window_size_ms, step_size_ms, path, title="", save_as='', raws=4, columns=3):
    """
    Calculate and plot the z-score normalized firing rate and its standard deviation for each dataset in the 
    flattened_data dictionary. Each dataset's firing rate is normalized and plotted in a subplot grid of 3x4.

    Parameters:
    - flattened_data: Dictionary with keys as dataset names and values as flattened spike times arrays.
    - window_size_ms: Size of the sliding window in milliseconds.
    - step_size_ms: Step size for the sliding window in milliseconds.
    - path: Output path to save the plot.
    - title: Title for the plot.
    - save_as: Filename suffix for the saved plot.
    """
    # Setting up the subplot grid
    fig, axs = plt.subplots(raws, columns, figsize=(15, 20), constrained_layout=True)
    fig.suptitle(f'{title} Z-Score Normalized Firing Rate: Window Size {window_size_ms}ms, Step Size {step_size_ms}ms', fontsize=16, fontweight='bold')

    # Calculate and plot z-score normalized firing rate for each dataset
    for i, (dataset_name, spike_times) in enumerate(flattened_data.items()):
        if i >= 12:  # To ensure it doesn't try to plot more than the grid size
            break
        
        start_time = np.min(spike_times)
        end_time = np.max(spike_times)
        window_size = window_size_ms / 1000  # Convert to seconds
        step_size = step_size_ms / 1000  # Convert to seconds
        times = np.arange(start_time, end_time - window_size, step_size)
        firing_rates = []

        for time in times:
            window_start = time
            window_end = time + window_size
            spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
            firing_rate = spikes_in_window / window_size
            firing_rates.append(firing_rate)

        mean_firing_rate = np.mean(firing_rates)
        std_firing_rate = np.std(firing_rates)
        z_scores = (firing_rates - mean_firing_rate) / std_firing_rate
        std_dev_z_scores = np.std(z_scores)
        
        # Plotting
        ax = axs.flatten()[i]
        ax.plot(times + window_size / 2, z_scores, linewidth=2, label='Z-Score')
        ax.fill_between(times + window_size / 2, z_scores - std_dev_z_scores, z_scores + std_dev_z_scores, color='gray', alpha=0.4, label='±1 SD')
        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Z-Score', fontsize=10)
        ax.legend()

    # Adjust subplots that are not used (if any)
    for j in range(i + 1, raws*columns):
        axs.flatten()[j].axis('off')

    # Save the plot
    if save_as:
        file_name = f"Subplots_{title}_z_firing_rate.png"
        plt.savefig(f"{path}/{file_name}", dpi=300)

    plt.show()
    print(f"Plot saved to {path}{file_name}")






def z_score_firing_rate_by_class_subplots(data, window_size_ms, step_size_ms, path, save_as='', ):
    """
    Calculate and plot the firing rate and its variability of neuronal data separated into six classes,
    each represented by a list within the data. Each list contains nested lists of spike times.
    This function processes each class, calculates the firing rate over a sliding square window,
    applies z-score normalization, and plots the results with a shadowed area representing 1 standard deviation
    around the mean z-score. Different colors represent different classes.

    Parameters:
    - data: List of six lists, each containing nested lists of spike times.
    - window_size_ms: Size of the sliding window in milliseconds.
    - step_size_ms: Step size for the sliding window in milliseconds.
    - path: Output path to save the plot.
    - save_as: Filename suffix for the saved plot.
    """
    colors = ['green', 'orange', 'cyan', 'magenta', 'blue', 'red']
    fig, axs = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f'Firing Rate: Window Size {window_size_ms}ms, Step Size {step_size_ms}ms', fontsize=16, fontweight='bold')

    for i, class_data in enumerate(data):
        spike_times = np.concatenate(class_data)
        start_time = min(spike_times)
        end_time = max(spike_times)
        window_size = window_size_ms / 1000
        step_size = step_size_ms / 1000
        times = np.arange(start_time, end_time - window_size, step_size)
        firing_rates = []

        for time in times:
            window_start = time
            window_end = time + window_size
            spikes_in_window = ((spike_times >= window_start) & (spike_times < window_end)).sum()
            firing_rate = spikes_in_window / window_size
            firing_rates.append(firing_rate)

        # Calculate z-score
        mean_firing_rate = np.mean(firing_rates)
        std_firing_rate = np.std(firing_rates)
        z_scores = (firing_rates - mean_firing_rate) / std_firing_rate
        std_dev_z_scores = np.std(z_scores)
        


        # Plot firing rate with z-score normalization
        ax = axs.flatten()[i]
        ax.plot(times + window_size / 2, z_scores, color=colors[i], linewidth=2, label='Z-score Normalized Firing Rate')
        # Shadowed area for +/- 1 standard deviation based on z-scores
        ax.fill_between(times + window_size / 2, z_scores - std_dev_z_scores, z_scores + std_dev_z_scores, color=colors[i], alpha=0.2, label='±1 SD Z-score')
        

        titles=["0", "6", "8", "10", "12", "24"]
        ax.set_title(f'Stimulus: {titles[i]} μm', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z-score Normalized Firing Rate')
        #ax.legend()

    # Save the plot
    if save_as:
        file_name= f"{save_as}_zscore_{window_size_ms}_{step_size_ms}.png"
        plt.savefig(f"{path}/{file_name}", dpi=300)

    plt.show()
    print(f"Plot saved to {path}{file_name}")










def z_score_firing_rate_by_class_single_plot(data, window_size_ms, step_size_ms, path, save_as=''):
    """
    Calculate and plot the firing rate and its variability of neuronal data separated into six classes,
    each represented by a list within the data. This function processes each class, calculates the firing rate
    over a sliding square window, applies z-score normalization, and plots the results with a shadowed area
    representing 1 standard deviation around the mean z-score. Different colors represent different classes
    on a single plot without showing standard deviation in the legend.

    Parameters:
    - data: List of six lists, each containing nested lists of spike times.
    - window_size_ms: Size of the sliding window in milliseconds.
    - step_size_ms: Step size for the sliding window in milliseconds.
    - path: Output path to save the plot.
    - save_as: Filename suffix for the saved plot.
    """
    colors = ['green', 'orange', 'cyan', 'magenta', 'blue', 'red']
    plt.figure(figsize=(15, 7))
    plt.title(f'Firing Rate and Variability by Class: Window Size {window_size_ms}ms, Step Size {step_size_ms}ms', fontsize=16, fontweight='bold')

    for i, class_data in enumerate(data):
        spike_times = np.concatenate(class_data)
        start_time = min(spike_times)
        end_time = max(spike_times)
        window_size = window_size_ms / 1000
        step_size = step_size_ms / 1000
        times = np.arange(start_time, end_time - window_size, step_size)
        firing_rates = []

        for time in times:
            window_start = time
            window_end = time + window_size
            spikes_in_window = ((spike_times >= window_start) & (spike_times < window_end)).sum()
            firing_rate = spikes_in_window / window_size
            firing_rates.append(firing_rate)

        # Calculate z-score
        mean_firing_rate = np.mean(firing_rates)
        std_firing_rate = np.std(firing_rates)
        z_scores = (firing_rates - mean_firing_rate) / std_firing_rate
        std_dev_z_scores = np.std(z_scores)

        # Plot firing rate with z-score normalization and shadowed area for +/- 1 standard deviation
        plt.plot(times + window_size / 2, z_scores, color=colors[i], alpha=0.5, linewidth=1, label=f'Class {i+1}')
        plt.fill_between(times + window_size / 2, z_scores - std_dev_z_scores, z_scores + std_dev_z_scores, color=colors[i], alpha=0.05)

    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Z-score Normalized Firing Rate', fontsize=12, fontweight='bold')
    plt.legend(loc='best', title="Classes")

    # Save the plot
    if save_as:
        file_name = f"Plot_{save_as.replace(' ', '_')}_z_score_{window_size_ms}_{step_size_ms}.png"
        plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
        
        
    plt.show()
    print(f"Plot saved to {path}/{file_name}")








def concatenate_plot_z_score_firing_rate(flattened_data, window_size_ms, step_size_ms, path, save_as=''):
    """
    Calculate and plot the z-score normalized firing rate of neuronal data from a dictionary of datasets,
    each represented by a flattened array of spike times. This function processes each dataset, calculates the firing rate
    over a sliding square window, applies z-score normalization, and plots the results on a single plot.
    Different colors are used to represent different datasets.

    Parameters:
    - flattened_data: Dictionary with keys as dataset names and values as flattened arrays of spike times.
    - window_size_ms: Size of the sliding window in milliseconds.
    - step_size_ms: Step size for the sliding window in milliseconds.
    - path: Output path to save the plot.
    - save_as: Filename suffix for the saved plot.
    """
    # Define a list of colors for visualization
    colors = ['green', 'orange', 'cyan', 'magenta', 'blue', 'brown', 'yellow', 'purple', 'red', 'grey']
    plt.figure(figsize=(15, 7))
    plt.title(f'Z-Score Normalized Firing Rate: Window Size {window_size_ms}ms, Step Size {step_size_ms}ms', fontsize=15, fontweight='bold')

    for i, (dataset_name, spike_times) in enumerate(flattened_data.items()):
        if i >= len(colors):  # Ensure we do not exceed the list of predefined colors
            break

        start_time = np.min(spike_times)
        end_time = np.max(spike_times)
        window_size = window_size_ms / 1000  # Convert to seconds
        step_size = step_size_ms / 1000  # Convert to seconds
        times = np.arange(start_time, end_time - window_size, step_size)
        firing_rates = []

        for time in times:
            window_start = time
            window_end = time + window_size
            spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
            firing_rate = spikes_in_window / window_size
            firing_rates.append(firing_rate)

        # Calculate z-score
        mean_firing_rate = np.mean(firing_rates)
        std_firing_rate = np.std(firing_rates)
        z_scores = (firing_rates - mean_firing_rate) / std_firing_rate
        std_dev_z_scores = np.std(z_scores)

        # Plot firing rate with z-score normalization
        plt.plot(times + window_size / 2, z_scores, color=colors[i], alpha=0.3, linewidth=2, label=f'{dataset_name}')
        plt.fill_between(times + window_size / 2, z_scores - std_dev_z_scores, z_scores + std_dev_z_scores, color=colors[i], alpha=0.0)

    plt.xlabel('Time (s)', fontsize=10, fontweight='bold')
    plt.ylabel('Z-Score Normalized Firing Rate', fontsize=10, fontweight='bold')
    plt.legend(loc='best', title="Datasets")

    # Save the plot
    if save_as:
        file_name = f"{save_as}_z_score_firing_rate.png"
        plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
        
    plt.show()
    print(f"Plot saved to {path}/{file_name}")
    
    
    


def concatenate_z_score_firing_rate(flattened_data, window_size_ms, step_size_ms, path, save_as=''):
    """
    Calculate and plot the z-score normalized firing rate of all neuronal data combined from a dictionary of datasets,
    each represented by a flattened array of spike times. This function concatenates all datasets, calculates the firing rate
    over a sliding square window, applies z-score normalization, and plots the results on a single plot.

    Parameters:
    - flattened_data: Dictionary with keys as dataset names and values as flattened arrays of spike times.
    - window_size_ms: Size of the sliding window in milliseconds.
    - step_size_ms: Step size for the sliding window in milliseconds.
    - path: Output path to save the plot.
    - save_as: Filename suffix for the saved plot.
    """
    # Concatenate all spike times from different datasets into one array
    all_spike_times = np.concatenate(list(flattened_data.values()))

    # Define plotting parameters
    plt.figure(figsize=(15, 7))
    plt.title(f'{save_as.replace("_", " ")} Z-Score Normalized Firing Rate: Window Size {window_size_ms}ms, Step Size {step_size_ms}ms', fontsize=13, fontweight='bold')

    # Calculate global start and end times
    start_time = np.min(all_spike_times)
    end_time = np.max(all_spike_times)
    window_size = window_size_ms / 1000  # Convert to seconds
    step_size = step_size_ms / 1000  # Convert to seconds
    times = np.arange(start_time, end_time - window_size, step_size)
    firing_rates = []

    # Calculate firing rates across all data
    for time in times:
        window_start = time
        window_end = time + window_size
        spikes_in_window = np.sum((all_spike_times >= window_start) & (all_spike_times < window_end))
        firing_rate = spikes_in_window / window_size
        firing_rates.append(firing_rate)

    # Calculate z-score
    mean_firing_rate = np.mean(firing_rates)
    std_firing_rate = np.std(firing_rates)
    z_scores = (firing_rates - mean_firing_rate) / std_firing_rate
    std_dev_z_scores = np.std(z_scores)

    # Plot firing rate with z-score normalization
    plt.plot(times + window_size / 2, z_scores, color='dodgerblue', alpha=0.9, linewidth=2, label='Combined Z-Scored FR')
    plt.fill_between(times + window_size / 2, z_scores - std_dev_z_scores, z_scores + std_dev_z_scores, color="lightblue", alpha=0.3)

    plt.xlabel('Time (s)', fontsize=10, fontweight='bold')
    plt.ylabel('Z-Score Normalized Firing Rate', fontsize=10, fontweight='bold')
    #plt.legend(loc='best', title="")

    # Save the plot
    if save_as:
        file_name = f"{save_as}_combined_z_score_firing_rate.png"
        plt.savefig(f"{path}/{file_name}", dpi=300, bbox_inches="tight")
        
    plt.show()
    print(f"Plot saved to {path}/{file_name}")


