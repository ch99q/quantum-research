from datetime import datetime
import os
import json
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, pauli_error
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
import numpy as np
import matplotlib.pyplot as plt
from experiment import (
    calculate_delays,
    create_bell_test_circuits,
    create_phase_test_circuits,
    create_ghz_test_circuits
)

def create_noise_model(t1=70000, t2=110000):  # Updated to Eagle's typical values
    """Create a realistic noise model matching IBM Eagle processor"""
    noise_model = NoiseModel()
    
    # Single qubit amplitude damping error (T1 decay)
    p_amp_damp = 1 - np.exp(-1/t1)
    amp_damp = amplitude_damping_error(p_amp_damp)
    noise_model.add_all_qubit_quantum_error(amp_damp, ['delay'])
    
    # Single-qubit gate error rates (Eagle's typical values)
    p_gate = 0.0003  # 0.03% average error rate for single-qubit gates
    gate_error = pauli_error([('X', p_gate), ('I', 1 - p_gate)])
    noise_model.add_all_qubit_quantum_error(gate_error, ['rx', 'rz'])
    
    # Two-qubit error for CX gates (Eagle's typical values)
    cx_error_rate = 0.008  # 0.8% average error rate for CNOT gates
    # Distribute error probability equally among error terms
    error_prob = cx_error_rate / 6  # Divide by number of error terms
    cx_error = pauli_error([
        ('IX', error_prob),
        ('IY', error_prob),
        ('IZ', error_prob),
        ('XI', error_prob),
        ('YI', error_prob),
        ('ZI', error_prob),
        ('II', 1 - (6 * error_prob))  # Remaining probability for identity
    ])
    noise_model.add_all_qubit_quantum_error(cx_error, ['cx'])
    
    return noise_model

def simulate_circuits(circuits, noise_model=None, shots=1024, method='density_matrix'):
    """Simulate a list of circuits with optional noise"""
    simulator = AerSimulator(
        method=method,
        noise_model=noise_model
    )
    
    results = []
    for circuit in circuits:
        try:
            # Transpile and run without debug prints
            circuit_transpiled = transpile(circuit, simulator)
            
            if method == 'density_matrix':
                circuit_transpiled.save_density_matrix(label='dm')
            elif method == 'statevector':
                circuit_transpiled.save_statevector(label='statevector')
            
            circuit_transpiled.barrier()
            job = simulator.run(circuit_transpiled, shots=shots)
            results.append(job.result())
            
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            continue
    
    return results

def analyze_bell_test_results(results, delays):
    """Analyze Bell state test results"""
    fidelities = []
    correlations = []
    
    # Create ideal Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    ideal_state = np.zeros((4, 4), dtype=complex)
    ideal_state[0,0] = ideal_state[0,3] = ideal_state[3,0] = ideal_state[3,3] = 0.5
    ideal_bell = DensityMatrix(ideal_state)
    
    for result in results:
        try:
            if 'dm' in result.data(0):
                # Get density matrix from result
                rho = DensityMatrix(result.data(0)['dm'])
                fidelity = state_fidelity(rho, ideal_bell)
            else:
                fidelity = None
            
            fidelities.append(fidelity)
            
            # Calculate ZZ correlation from counts
            counts = result.get_counts()
            correlation = calculate_correlation(counts)
            correlations.append(correlation)
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            fidelities.append(None)
            correlations.append(None)
    
    return fidelities, correlations

def analyze_phase_test_results(results, delays):
    """Analyze phase coherence test results"""
    coherences = []
    purities = []
    
    for result in results:
        try:
            # Get density matrix from result using correct key 'dm'
            rho = DensityMatrix(result.data(0)['dm'])
            
            # Calculate coherence (off-diagonal elements)
            coherence = 2 * abs(rho.data[0][1])
            coherences.append(coherence)
            
            # Calculate purity
            purity = float(np.real(np.trace(rho.data @ rho.data)))
            purities.append(purity)
            
        except Exception as e:
            print(f"Error analyzing phase result: {e}")
            coherences.append(None)
            purities.append(None)
    
    return coherences, purities

def analyze_ghz_test_results(results, delays):
    """Analyze GHZ state test results"""
    fidelities = []
    witnesses = []
    
    for result in results:
        try:
            if 'statevector' in result.data(0):
                # Convert statevector to density matrix using outer product
                sv = Statevector(result.data(0)['statevector'])
                state_vector = sv.data
                state = DensityMatrix(np.outer(state_vector, np.conj(state_vector)))
            elif 'dm' in result.data(0):
                state = DensityMatrix(result.data(0)['dm'])
            else:
                # Use counts to estimate state
                counts = result.get_counts()
                n_qubits = len(next(iter(counts)))
                state = counts_to_density_matrix(counts, n_qubits)
            
            n_qubits = int(np.log2(len(state.data)))
            
            # Create ideal GHZ state for comparison
            ideal_ghz = create_ideal_ghz_state(n_qubits)
            
            # Calculate fidelity
            fidelity = state_fidelity(state, ideal_ghz)
            fidelities.append(fidelity)
            
            # Calculate entanglement witness
            witness = calculate_ghz_witness(state, n_qubits)
            witnesses.append(witness)
            
        except Exception as e:
            print(f"Error analyzing GHZ result: {str(e)}")
            print(f"Result data keys: {result.data(0).keys()}")
            fidelities.append(None)
            witnesses.append(None)
    
    return fidelities, witnesses

def counts_to_density_matrix(counts, n_qubits):
    """Convert measurement counts to an estimated density matrix"""
    dim = 2**n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    total_counts = sum(counts.values())
    
    # Diagonal elements from measurement probabilities
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        rho[idx, idx] = count / total_counts
    
    return DensityMatrix(rho)

def calculate_correlation(counts):
    """Calculate the correlation value from measurement counts"""
    corr = 0
    total = sum(counts.values())
    for outcome, count in counts.items():
        # Convert outcome to ±1 correlations
        value = 1 if outcome.count('1') % 2 == 0 else -1
        corr += value * count / total
    return corr

def create_ideal_ghz_state(n_qubits):
    """Create ideal GHZ state density matrix"""
    # Create GHZ statevector
    dim = 2**n_qubits
    state_vector = np.zeros(dim, dtype=complex)
    state_vector[0] = 1/np.sqrt(2)
    state_vector[-1] = 1/np.sqrt(2)
    
    # Convert to density matrix using outer product
    return DensityMatrix(np.outer(state_vector, np.conj(state_vector)))

def calculate_ghz_witness(rho, n_qubits):
    """Calculate GHZ state entanglement witness"""
    ideal_ghz = create_ideal_ghz_state(n_qubits)
    witness = 0.5 - float(np.real(state_fidelity(rho, ideal_ghz)))
    return witness

def plot_results(delays, results_dict, title, ax=None, params=None):
    """Plot results over time with parameter details"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot configurations with theoretical limits
    plot_configs = {
        'Bell State Analysis': {
            'limits': {
                'Classical Limit': 0.7071,  # 1/√2
                'Heisenberg Limit': 1.0
            },
            'fill_region': (0, 0.7071)
        },
        'Phase Coherence Analysis': {
            'limits': {
                'Random Phase': 0.5,
                'Perfect Coherence': 1.0
            }
        },
        'GHZ State Analysis': {
            'limits': {
                'Classical Bound': 0.5,
                'Genuine Entanglement': 0.5,
                'Perfect GHZ': 1.0
            },
            'fill_region': (0, 0.5)
        }
    }
    
    config = plot_configs.get(title, {})
    
    valid_data = False
    for label, values in results_dict.items():
        # Filter out None values and convert to numpy array
        valid_points = [(d, v) for d, v in zip(delays, values) if v is not None]
        if valid_points:
            d, v = zip(*valid_points)
            d = np.array(d)/1000  # Convert to μs
            v = np.array(v)
            
            # Add small offset to zero values for log scale
            if np.any(v == 0):
                min_nonzero = np.min(v[v > 0]) if np.any(v > 0) else 1e-10
                v[v == 0] = min_nonzero/10
            
            ax.plot(d, v, 'o-', label=label)
            valid_data = True
    
    if valid_data:
        # Add theoretical limits
        colors = ['r', 'g', 'b']
        styles = ['--', ':', '-.']
        for (limit_name, limit_value), color, style in zip(
            config.get('limits', {}).items(), colors, styles
        ):
            ax.axhline(y=limit_value, color=color, linestyle=style,
                      label=limit_name, alpha=0.5)
        
        # Add shaded regions for classical bounds
        if 'fill_region' in config:
            start, end = config['fill_region']
            ax.fill_between(ax.get_xlim(), start, end,
                          color='red', alpha=0.1, label='Classical Region')
        
        ax.set_xscale('log')
        ax.set_xlabel('Delay (μs)')
        ax.set_ylabel('Value')
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        
        # Add parameter text box if provided
        if params:
            param_text = (
                f"T1={params['t1']/1000:.0f}μs\n"
                f"T2={params['t2']/1000:.0f}μs\n"
                f"1Q Error: {params['p_gate']*100:.3f}%\n"
                f"2Q Error: {params['cx_error_rate']*100:.3f}%\n"
                f"Shots: {params['shots']}"
            )
            ax.text(0.02, 0.98, param_text,
                   transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                   verticalalignment='top',
                   fontsize=8)
        
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        print(f"No valid data to plot for {title}")

def print_experiment_summary(name, delays, results_dict):
    """Print a summary of experimental results"""
    print(f"\n{name} Test Summary:")
    print("-" * 50)
    
    for metric, values in results_dict.items():
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            print(f"{metric}: No valid data")
            continue
            
        print(f"\n{metric}:")
        print(f"  Initial value: {valid_values[0]:.4f}")
        print(f"  Final value: {valid_values[-1]:.4f}")
        print(f"  Maximum: {max(valid_values):.4f}")
        print(f"  Minimum: {min(valid_values):.4f}")  # Fixed format string
        print(f"  Mean: {np.mean(valid_values):.4f}")
        print(f"  Std dev: {np.std(valid_values):.4f}")
        
        # Calculate decay time (time to reach 1/e of initial value)
        if valid_values[0] > valid_values[-1]:
            decay_times = []
            threshold = valid_values[0] / np.e
            for d, v in zip(delays, values):
                if v is not None and v < threshold:
                    decay_times.append(d)
                    break
            if decay_times:
                print(f"  T1/e decay time: {decay_times[0]/1000:.1f} μs")
        print()  # Add blank line between metrics

def main():
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup simulation parameters
    delays = calculate_delays()
    qubit_line = list(range(7))
    
    # Define all simulation parameters
    sim_params = {
        't1': 70000,
        't2': 110000,
        'p_gate': 0.0003,
        'cx_error_rate': 0.008,
        'shots': 1024
    }
    
    noise_model = create_noise_model(t1=sim_params['t1'], t2=sim_params['t2'])
    
    # Create figure for all plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    all_results = {}
    
    # Create and run circuits
    for idx, (name, creator_func) in enumerate([
        ('Bell', create_bell_test_circuits),
        ('Phase', create_phase_test_circuits),
        ('GHZ', create_ghz_test_circuits)
    ]):
        print(f"\nSimulating {name} test...")
        circuits = creator_func(qubit_line, delays)
        
        method = 'density_matrix' if name != 'GHZ' else 'statevector'
        sim_results = simulate_circuits(
            circuits, 
            noise_model=(None if name == 'GHZ' else noise_model),
            method=method,
            shots=8192 if method == 'statevector' else 1024
        )
        
        # Analyze results
        if name == 'Bell':
            fidelities, correlations = analyze_bell_test_results(sim_results, delays)
            results_dict = {
                'Fidelity': fidelities,
                'Correlation': correlations
            }
        elif name == 'Phase':
            coherences, purities = analyze_phase_test_results(sim_results, delays)
            results_dict = {
                'Coherence': coherences,
                'Purity': purities
            }
        else:  # GHZ
            fidelities, witnesses = analyze_ghz_test_results(sim_results, delays)
            results_dict = {
                'Fidelity': fidelities,
                'Entanglement Witness': witnesses
            }
        
        # Plot and save results
        plot_results(delays, results_dict, f'{name} State Analysis', 
                    axes[idx], params=sim_params)
        print_experiment_summary(name, delays, results_dict)
        
        # Store results for JSON
        all_results[name] = {
            'delays': delays,
            'results': results_dict
        }
    
    # Adjust layout and save combined plot
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'simulation_results_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to JSON
    json_results = {
        'timestamp': timestamp,
        'parameters': sim_params,
        'data': all_results
    }
    
    json_path = os.path.join(results_dir, f'simulation_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\nSimulation complete! Results saved to:")
    print(f"- Plots: {plot_path}")
    print(f"- Data: {json_path}")

if __name__ == "__main__":
    main()