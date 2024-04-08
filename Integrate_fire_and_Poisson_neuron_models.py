import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, 100, 1000)  # 100 milliseconds
dt = time[1] - time[0]
threshold = 1
potential = np.zeros_like(time)
input_current = 0.02  # Constant input current

for i in range(1, len(time)):
    potential[i] = potential[i-1] + input_current
    if potential[i] >= threshold:
        potential[i] = 0  # Reset potential

plt.plot(time, potential)
plt.title('Integrate-and-Fire Model')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential')
plt.show()






rate = 50  # Spikes per second
duration = 1  # Second
spike_times = np.random.poisson(lam=rate, size=rate*duration)

plt.eventplot(spike_times, color="black")
plt.title('Traditional Poisson Model')
plt.xlabel('Time (ms)')
plt.ylabel('Spike Train')
plt.ylim(0, 1)  # Adjust for better visualization
plt.show()









time = np.linspace(0, 100, 1000)  # 100 milliseconds
signal = np.full_like(time, fill_value=1)  # Constant signal
noise_variance = 0.5
noise = np.random.normal(0, np.sqrt(noise_variance), size=time.shape)
firing_rate = 1 / (1 + np.exp(-(signal + noise)))  # Sigmoid function as an example

spikes = np.random.rand(len(time)) < firing_rate  # Determine spikes based on firing rate
plt.eventplot(time[spikes], color="black", linewidth=0.4)
plt.title('Modified Poisson Model')
plt.xlabel('Time (ms)')
plt.ylabel('Spike Train')
plt.ylim(0, 1)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Time vector
time = np.linspace(0, 1, 1000)  # 1 second
# Example signal and noise
signal = np.ones_like(time)  # Constant signal
noise_variance = 0.1
noise = np.random.normal(0, np.sqrt(noise_variance), size=time.shape)

# Firing probability using erf
firing_probability = erf(signal + noise)

# Plot firing probability
plt.plot(time, firing_probability)
plt.title('Firing Probability Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Firing Probability')
plt.show()


"""
The plot generated from your code, depicting the firing probability over time for a neuron modeled using the modified Poisson framework with the error function (erf), offers a dynamic view of how a neuron's probability of firing changes in response to combined deterministic signals and stochastic noise. Here's how to interpret the plot:

### Understanding the Axes

- **X-axis (Time)**: Represents the observation period, from the start to the end of the simulation (in your case, 1 second). Each point along this axis corresponds to a specific moment within the simulation timeframe.
  
- **Y-axis (Firing Probability)**: Shows the neuron's firing probability at each point in time. The firing probability is the likelihood that the neuron will fire at a given moment, ranging from -1 to 1 due to the use of the erf function. In practical applications for firing probabilities, you would often normalize or transform this range to 0 to 1, where 0 indicates no chance of firing and 1 represents certainty of firing.

### Interpreting the Curve

- **Shape and Fluctuations**: The curve's shape illustrates how the neuron's firing probability responds to the input signal and noise over time. A rising section of the curve suggests increasing firing probability, whereas a falling section indicates decreasing probability. The smoothness or variability of the curve reflects the interplay between the signal and noise—how deterministic inputs and random fluctuations combine to influence the neuron's behavior.

- **Impact of Noise**: The presence of noise in the model introduces variability into the firing probability. Even with a constant signal (as in your example), the noise can cause the firing probability to fluctuate, simulating the variability observed in real neuronal activity. Points on the curve where the firing probability changes more sharply might indicate moments when the noise significantly augments or diminishes the effect of the signal.

### Practical Implications

- **Modeling Real Neuronal Dynamics**: This plot provides insight into how real neurons might respond to a mix of deterministic stimuli and stochastic environmental noise. Neurons in the brain rarely operate under constant conditions; instead, they continuously integrate varying inputs against a backdrop of inherent noise. The plot captures this dynamic, showing that neuronal firing probability is not static but varies in response to ongoing changes in inputs.

- **Predicting Neuronal Behavior**: By examining the firing probability over time, researchers can make predictions about when a neuron is more or less likely to fire. For instance, periods where the firing probability is higher suggest moments of increased neural activity, potentially in response to external stimuli or internal processes.

### Conclusion

The plot serves as a visual representation of the nuanced and probabilistic nature of neuronal firing, highlighting the complex interplay between deterministic inputs and stochastic influences. It underscores the value of the modified Poisson model in capturing more realistic neuronal behaviors compared to simpler models that might assume constant firing rates or ignore the impact of noise. By analyzing such plots, researchers can gain deeper insights into the mechanisms underlying neural computation and the factors that modulate neuronal activity.


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Shared simulation parameters
duration = 1  # in seconds
time = np.linspace(0, duration, 1000)  # 1000 time points over 1 second

# Traditional Poisson Model Parameters
rate = 50  # Average firing rate in spikes per second

# Generate spikes for the Traditional Poisson Model
spike_counts = np.random.poisson(lam=rate * duration)
spike_times_traditional = np.sort(np.random.uniform(0, duration, spike_counts))

# Modified Poisson Model Parameters
signal = 0.5  # Constant signal strength
noise_variance = 0.2  # Variance of noise
noise = np.random.normal(0, np.sqrt(noise_variance), size=time.shape)

# Sigmoid function for firing probability; using erf for demonstration
firing_prob = (1 + erf(signal + noise)) / 2  # Normalizing to range [0, 1]
spike_probabilities = np.random.uniform(0, 1, size=time.shape)
spikes_modified = time[spike_probabilities < firing_prob]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharex=True)

# Traditional Poisson Model Plot
axs[0].eventplot(spike_times_traditional, color="black")
axs[0].set_title('Traditional Poisson Model')
axs[0].set_ylabel('Spike Train')

# Modified Poisson Model Plot
axs[1].eventplot(spikes_modified, color="black", linewidth=0.5)
axs[1].set_title('Modified Poisson Model')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Spike Train')

plt.tight_layout()
plt.show()







plt.figure(figsize=(10, 4))
plt.plot(time, firing_prob, label='Firing Probability')
plt.xlabel('Time (s)')
plt.ylabel('Probability')
plt.title('Firing Probability Over Time (Modified Poisson Model)')
plt.legend()
plt.show()


