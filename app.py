#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the processed WWTW data
wwtw_data = pd.read_csv('processed_wwtw_data.csv')

# Load HSdata to get the list of local authorities and their distributions
hs_data = pd.read_csv('HSdata.csv')
local_authorities = hs_data['Local_authority'].unique()

def compute_B(N, Oc, W, C, L, Cl, Ee, A, En):
    return ((N * Oc * W * C * (3.65e-4)) + (N * Oc * L * Cl * (3.65e-4)) - (Ee * A) + (En * A))

def run_monte_carlo_simulation (N, Oc, W, C, L, Cl, Ee, A, En, std_dev_oc):
    # Updated means dictionary to use the bootstrapped Oc value
    means = {"N": N, "Oc": Oc, "W": W, "C": C, "L": L, "Cl": Cl, "Ee": Ee, "A": A, "En": En}
    
    # Updated std_devs dictionary to use the bootstrapped standard deviation of Oc
    std_devs = {"N": 0, "Oc": std_dev_oc, "W": 5, "C": 0.19, "L": 10, "Cl": 0.15, "Ee": 0.01, "A": 0.1, "En": 0.3}

    n_simulations = 100000
    outputs = []

    for _ in range(n_simulations):
        sampled_values = {var: np.random.normal(mean, std_devs[var]) for var, mean in means.items()}
        output = compute_B(**sampled_values)
        outputs.append(output)
   
    results = {
        'mean_output': np.mean(outputs),
        'median_output': np.median(outputs),
        'std_dev_output': np.std(outputs)
    }

    plt.figure()
    plt.hist(outputs, bins=50, color='dimgray', edgecolor='k')
    plt.xlabel('Nutrient budget (kg/yr)')
    plt.ylabel('Frequency')
    plt.title('Nutrient Budget Simulation Results')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return results, f"data:image/png;base64,{plot_url}"


def bootstrap_oc(dist, N, num_trials=1000):
    means = []
    for _ in range(num_trials):
        samples = np.random.choice(dist, size=N, replace=True)
        sample_mean = np.mean(samples)
        means.append(sample_mean)
    
    # Calculate the mean of means and the standard deviation of these means
    mean_oc = np.mean(means)
    std_dev_oc = np.std(means)
    return mean_oc, std_dev_oc

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_authority = request.form.get('Local_authority')
        N = int(request.form.get('N'))  

        authority_data = hs_data[hs_data['Local_authority'] == selected_authority]
        population_dist = np.repeat(authority_data['household_size'].values, authority_data['Frequency'].values)

        mean_oc, std_dev_oc = bootstrap_oc(population_dist, N)

        return render_template('index.html', local_authorities=local_authorities, selected_authority=selected_authority, wwtw_data=wwtw_data.to_dict('records'), result=True)
    else:
        return render_template('index.html', local_authorities=local_authorities, wwtw_data=wwtw_data.to_dict('records'))

@app.route('/get_oc_values', methods=['POST'])
def get_oc_values():
    selected_authority = request.form.get('Local_authority')
    N = int(request.form.get('N', 100))  # Default sample size if not provided

    authority_data = hs_data[hs_data['Local_authority'] == selected_authority]
    population_dist = np.repeat(authority_data['household_size'].values, authority_data['Frequency'].values)

    mean_oc, std_dev_oc = bootstrap_oc(population_dist, N)

    return jsonify({'mean_oc': mean_oc, 'std_dev_oc': std_dev_oc})

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        # Extract parameters from the request
        N = float(request.form['N'])
        Oc = float(request.form['Oc'])
        W = float(request.form['W'])
        C = float(request.form['C'])
        L = float(request.form['L'])
        Cl = float(request.form['Cl'])
        Ee = float(request.form['Ee'])
        A = float(request.form['A'])
        En = float(request.form['En'])
        std_dev_oc = float(request.form.get('std_dev_oc', 0.09))
        
          # Print the inputs received for debugging
        print("Received inputs for Monte Carlo Simulation:")
        print(f"N: {N}, Oc: {Oc}, std_dev_oc: {std_dev_oc}, W: {W}, C: {C}, L: {L}, Cl: {Cl}, Ee: {Ee}, A: {A}, En: {En}")

        # Run Monte Carlo simulation with the received values
        results, graph_data = run_monte_carlo_simulation (N, Oc, W, C, L, Cl, Ee, A, En, std_dev_oc)

        # Return the results and graph data as JSON
        return jsonify({'stats': results, 'graph': graph_data})
    
        print("Received form data:", request.form)

    except Exception as e:
        # Log the error and return an error message
        print("An error occurred during the simulation:", str(e))
        return jsonify({'error': 'An error occurred during the simulation.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
