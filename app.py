#!/usr/bin/env python
# coding: utf-8

# In[10]:


from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Your Monte Carlo simulation functions
def compute_B(N, Oc, W, C, L, Cl, Ee, A, En):
    return ((N * Oc * W * C * (3.65e-4)) + (N * Oc * L * Cl * (3.65e-4)) - (Ee * A) + (En * A))

def run_monte_carlo_simulation(N, Oc, W, C, L, Cl, Ee, A, En):
    means = {"N": N, "Oc": Oc, "W": W, "C": C, "L": L, "Cl": Cl, "Ee": Ee, "A": A, "En": En}
    std_devs = {"N": 0, "Oc": 0.09, "W": 5, "C": 0.19, "L": 10, "Cl": 0.15, "Ee": 0.01, "A": 0.1, "En": 0.3}

    n_simulations = 100000
    outputs = []

    for _ in range(n_simulations):
        sampled_values = {var: np.random.normal(mean, std_devs[var]) for var, mean in means.items()}
        output = compute_B(**sampled_values)
        outputs.append(output)
   
    # Statistical Summary
    results = {
        'mean_output': np.mean(outputs),
        'median_output': np.median(outputs),
        'std_dev_output': np.std(outputs)
       }

    # Generate the histogram plot

    plt.hist(outputs, bins=50, color='dimgray', edgecolor='k')
    plt.xlabel('Nutrient budget (kg/yr)')
    plt.ylabel('Frequency')
    plt.title('Nutrient Budget Simulation Results')
    plt.tight_layout()

    # Convert plot to a PNG image in base64 format
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return results, f"data:image/png;base64,{plot_url}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    # Extracting parameters from the submitted form
    N = float(request.form['N'])
    Oc = float(request.form['Oc'])
    W = float(request.form['W'])
    C = float(request.form['C'])
    L = float(request.form['L'])
    Cl = float(request.form['Cl'])
    Ee = float(request.form['Ee'])
    A = float(request.form['A'])
    En = float(request.form['En'])

    # Run Monte Carlo simulation
    results, graph_data = run_monte_carlo_simulation(N, Oc, W, C, L, Cl, Ee, A, En)

    # Return the results and graph data
    return jsonify({'stats': results, 'graph': graph_data})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




