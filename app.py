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

def run_monte_carlo_simulation(N, Oc, W, C, L, Cl, Ee, A, En, std_dev_oc, st_dev_ee, st_dev_en, std_dev_w):
    means = {"N": N, "Oc": Oc, "W": W, "C": C, "L": L, "Cl": Cl, "Ee": Ee, "A": A, "En": En}
    std_devs = {"N": 0, "Oc": std_dev_oc, "W": std_dev_w, "C": 0.19, "L": 10, "Cl": 0, "Ee": st_dev_ee, "A": 0.1, "En": st_dev_en}
    outputs = []
    for _ in range(100000):
        sampled_values = {var: np.random.normal(mean, std_devs[var]) for var, mean in means.items()}
        output = compute_B(**sampled_values)
        outputs.append(output)
    results = {'mean_output': np.mean(outputs), 'median_output': np.median(outputs), 'std_dev_output': np.std(outputs)}

    img = io.BytesIO()
    plt.figure(figsize=(8, 4), dpi=300)
    plt.hist(outputs, bins=50, color='#6dd5ed', edgecolor='k')  # Adjusted to match website's color
    plt.axvline(np.mean(outputs), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(outputs):.2f} kg/yr')
    plt.axvline(np.percentile(outputs, 5), color='orange', linestyle='dashed', linewidth=1, label=f'5th Percentile: {np.percentile(outputs, 5):.2f} kg/yr')
    plt.axvline(np.percentile(outputs, 95), color='orange', linestyle='dashed', linewidth=1, label=f'95th Percentile: {np.percentile(outputs, 95):.2f} kg/yr')
    plt.xlabel('Nutrient budget (kg/yr)')
    plt.ylabel('Frequency')
    plt.title('Nutrient Budget Simulation Results')
    plt.legend()
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    formatted_results = {
        'mean_output': f"Mean Nutrient Budget: {results['mean_output']:.3f} kg/yr",
        'median_output': f"Median Nutrient Budget: {results['median_output']:.3f} kg/yr",
        'std_dev_output': f"Standard Deviation of Nutrient Budget: {results['std_dev_output']:.3f} kg/yr"
    }

    return formatted_results, f"data:image/png;base64,{plot_url}"


def bootstrap_oc(dist, N, num_trials=1000):
    means = []
    for _ in range(num_trials):
        samples = np.random.choice(dist, size=N, replace=True)
        means.append(np.mean(samples))
    return np.mean(means), np.std(means)

def calculate_std_dev_w(N):
    mean_w = 113.7  # Mean of per capita water consumption
    sample_std_dev = 22.65  # Standard deviation for W
    num_samples = 100000  # Number of samples for bootstrapping

    # Generate samples based on normal distribution
    samples = np.random.normal(mean_w, sample_std_dev, (num_samples, N))
    sample_means = np.mean(samples, axis=1)
    
    # Calculate the standard deviation of the sampled means
    std_dev_w = np.std(sample_means)
    return std_dev_w


def sensitivity_analysis(means, std_devs):
    avg_changes = {}
    base_B = compute_B(**means)
    for var in means:
        perturbations = np.random.uniform(means[var] - 2*std_devs[var], means[var] + 2*std_devs[var], 1000)
        total_change = sum(abs(compute_B(**{**means, **{var: pert}}) - base_B) for pert in perturbations)
        avg_changes[var] = total_change / 1000
    total_avg_change = sum(avg_changes.values())
    return {var: change / total_avg_change for var, change in avg_changes.items()}


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

@app.route('/simulate_w', methods=['POST'])
def simulate_w():
    try:
        
        N = int(request.form.get('N', 100))  # Default to 100 if N is not provided
        std_dev_w = calculate_std_dev_w(N)
        mean_w = 113.7  

        # Return the calculated mean and standard deviation of W to the frontend
        return jsonify({'mean_w': mean_w, 'std_dev_w': std_dev_w})

    except Exception as e:
        print("An error occurred during the W simulation:", str(e))
        return jsonify({'error': 'An error occurred during the W simulation.'}), 500


@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        # Extract parameters from the request
        N = float(request.form['N'])
        Oc = float(request.form['Oc'])
        W = float(request.form['W'])
        C = float(request.form['C'])
        L = 0.22 * W
        Cl = 1
        Ee = float(request.form['Ee'])
        A = float(request.form['A'])
        En = float(request.form['En'])
        std_dev_oc = float(request.form.get('std_dev_oc', 0.09))
        st_dev_ee = 0.15 * Ee
        st_dev_en = 0.15 * En
        std_dev_w = float(request.form['std_dev_w'])

        
          # Print the inputs received for debugging
        print("Received inputs for Monte Carlo Simulation:")
        print(f"N: {N}, Oc: {Oc}, std_dev_oc: {std_dev_oc}, W: {W}, C: {C}, L: {L}, Cl: {Cl}, Ee: {Ee}, A: {A}, En: {En}, st_dev_ee: {st_dev_ee}, st_dev_en: {st_dev_en}, std_dev_w: {std_dev_w}")

        # Run Monte Carlo simulation with the received values
        results, graph_data = run_monte_carlo_simulation (N, Oc, W, C, L, Cl, Ee, A, En, std_dev_oc, st_dev_ee, st_dev_en, std_dev_w)

        # Return the results and graph data as JSON
        return jsonify({'stats': results, 'graph': graph_data})
    
        print("Received form data:", request.form)
        
    except Exception as e:
        # Log the error and return an error message
        print("An error occurred during the simulation:", str(e))
        return jsonify({'error': 'An error occurred during the simulation.'}), 500

@app.route('/sensitivity_analysis', methods=['POST'])
def run_sensitivity_analysis():
    try:
        # Retrieve JSON data from request
        data = request.json

        # Extract parameters from JSON
        N = float(data['N'])
        Oc = float(data['Oc'])
        W = float(data['W'])
        C = float(data['C'])
        L = 0.22 * W  # L is defined as 0.22 times W
        Cl = 1         # Cl is fixed at 1
        Ee = float(data['Ee'])
        A = float(data['A'])
        En = float(data['En'])
        std_dev_oc = float(data['std_dev_oc'])
        st_dev_ee = 0.15 * Ee
        st_dev_en = 0.15 * En
        std_dev_w = float(data['std_dev_w'])

        # Define the means and standard deviations for the simulation
        means = {"N": N, "Oc": Oc, "W": W, "C": C, "L": L, "Cl": Cl, "Ee": Ee, "A": A, "En": En}
        std_devs = {"N": 0, "Oc": std_dev_oc, "W": std_dev_w, "C": 0.19, "L": 10, "Cl": 0, "Ee": st_dev_ee, "A": 0.1, "En": st_dev_en}

        # Perform sensitivity analysis
        sensitivity_indices = sensitivity_analysis(means, std_devs)

        # Generate and return the sensitivity analysis graph
        buf = io.BytesIO()
        plt.figure(figsize=(8, 4), dpi=300)
        plt.bar(list(sensitivity_indices.keys()), list(sensitivity_indices.values()), color='#6dd5ed', edgecolor='k', zorder=2)
        plt.ylabel('Sensitivity Index', fontsize=14)
        plt.grid(axis='y', linestyle='-', alpha=0.7, color='darkgrey', zorder=1)
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        sensitivity_graph = base64.b64encode(buf.getvalue()).decode()

        return jsonify({'sensitivity_graph': sensitivity_graph})

    except Exception as e:
        # Log the error and return an error message
        print("An error occurred during the sensitivity analysis:", str(e))
        return jsonify({'error': 'An error occurred during the sensitivity analysis.'}), 500

   

if __name__ == '__main__':
    app.run(debug=True)
