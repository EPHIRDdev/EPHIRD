#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def compute_B(N, Oc, W, C, L, Cl, Ee, A, En):
    return ((N * Oc * W * C * (3.65e-4)) + (N * Oc * L * Cl * (3.65e-4)) - (Ee * A) + (En * A))

means = {
    "N": 194,
    "Oc": 2.32,
    "W": 111,
    "C": 0.44,
    "L": 25,
    "Cl": 1,
    "Ee": 0.02,
    "A": 9.3,
    "En": 1.65
}

std_devs = {
    "N": 0,
    "Oc": 0.09,
    "W": 5,
    "C": 0.19,
    "L": 10,
    "Cl": 0.15,
    "Ee": 0.01,
    "A": 0.1,
    "En": 0.3
}

n_simulations = 100000
outputs = []

for _ in range(n_simulations):
    sampled_values = {var: np.random.normal(mean, std_devs[var]) for var, mean in means.items()}
    output = compute_B(**sampled_values)
    outputs.append(output)

def compute_C(N, Oc, W, C, Ee, A, En):
    return ((N * Oc * W * C * 0.9 * (3.65e-4)) - (Ee * A) + (En * A))

specific_output = compute_C(194, 2.4, 120, 0.8, 0.02, 9.3, 1.65)
specific_output_times_1_2 = specific_output * 1.2

mean_output = np.mean(outputs)
median_output = np.median(outputs)
std_dev_output = np.std(outputs)
percentiles = np.percentile(outputs, [5, 25, 75, 95, 2.5, 97.5])

print(f"Mean of Output: {mean_output:.2f}")
print(f"Median of Output: {median_output:.2f}")
print(f"Standard Deviation of Output: {std_dev_output:.2f}")
print(f"5th Percentile: {percentiles[0]:.2f}")
print(f"25th Percentile: {percentiles[1]:.2f}")
print(f"75th Percentile: {percentiles[2]:.2f}")
print(f"95th Percentile: {percentiles[3]:.2f}")
print(f"calculator: {specific_output_times_1_2:.2f}")

critical_threshold = specific_output_times_1_2
probability_exceeding_threshold = np.sum(np.array(outputs) > critical_threshold) / n_simulations
print(f"Probability of Output Exceeding {critical_threshold}: {probability_exceeding_threshold:.4f}")

plt.rcParams['font.family'] = 'STIXGeneral'
plt.figure(figsize=(10, 6), dpi=300)
plt.hist(outputs, bins=50, color='dimgray', edgecolor='k', alpha=1.0, zorder=2)
plt.axvline(specific_output_times_1_2, color='red', linestyle='--', linewidth=2, label=f"Nutrient Budget Calculator with buffer")
plt.axvline(specific_output, color='blue', linestyle='--', linewidth=2, label=f"Nutrient Budget Calculator without buffer")
plt.axvline(percentiles[4], color='black', linestyle='--', linewidth=1, label=f"95% simulations")
plt.axvline(percentiles[5], color='black', linestyle='--', linewidth=1)

plt.xlabel('Nutrient Budget (kg/yr)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(loc="upper right")
plt.grid(axis='both', linestyle='-', linewidth=0.5, alpha=0.5, color='darkgray', zorder=1)
plt.tight_layout()
plt.show()

