from qiskit_ibm_runtime import QiskitRuntimeService
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from dotenv import load_dotenv
from collections import Counter

def load_quantum_config():
    """Load environment variables for quantum computing"""
    load_dotenv()
    IBMQ_TOKEN = os.getenv('IBMQ_TOKEN')
    CHANNEL = os.getenv('CHANNEL', 'ibm_quantum')
    INSTANCE = os.getenv('INSTANCE', 'ibm-q/open/main')
    
    if not IBMQ_TOKEN:
        raise ValueError("IBMQ_TOKEN not found in environment variables")
    
    return {
        'token': IBMQ_TOKEN,
        'channel': CHANNEL,
        'instance': INSTANCE
    }

def get_latest_job_info():
    """Get the latest job information"""
    latest_path = 'jobs/latest.json'
    if not os.path.exists(latest_path):
        raise FileNotFoundError("No job information found")
    
    with open(latest_path, 'r') as f:
        return json.load(f)

def remap_counts(counts, qubit_line):
    """Remap counts from physical to logical qubits"""
    physical_to_logical = {phys: log for log, phys in enumerate(qubit_line)}
    remapped_counts = {}
    
    for bitstring, count in counts.items():
        remapped_bits = ['0'] * len(qubit_line)
        for phys_pos, bit in enumerate(bitstring):
            if phys_pos in physical_to_logical:
                logical_pos = physical_to_logical[phys_pos]
                remapped_bits[logical_pos] = bit
        remapped_key = ''.join(remapped_bits)
        remapped_counts[remapped_key] = remapped_counts.get(remapped_key, 0) + count
    
    return remapped_counts

def bitarray_to_counts(bitarray):
    """Convert BitArray data to counts dictionary"""
    counts = {}
    try:
        # First attempt: try accessing raw array data
        raw_array = bitarray._array
        for shot_idx in range(bitarray.num_shots):
            shot_value = raw_array[shot_idx][0]
            key = format(shot_value, f'0{bitarray.num_bits}b')
            counts[key] = counts.get(key, 0) + 1
    except AttributeError:
        # Fallback: iterate through bits directly
        for shot_idx in range(bitarray.num_shots):
            shot = []
            for bit_idx in range(bitarray.num_bits):
                val = bitarray[shot_idx, bit_idx]
                shot.append(str(int(val)))
            bitstring = ''.join(shot)
            counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts

def analyze_quantum_results(results, qubit_line, test_type, circuit_indices):
    """Analyze quantum results based on test type"""
    start_idx, end_idx = circuit_indices[test_type]
    analyzed_data = []
    
    for circuit_result in results[start_idx:end_idx]:
        # Get measurement data from data registers
        if hasattr(circuit_result, 'data'):
            # Try to find any measurement register
            bitarray = None
            data_attrs = dir(circuit_result.data)
            
            # Find all measurement registers (c0, c1, c2, etc)
            measurement_regs = [attr for attr in data_attrs if attr.startswith('c') and attr[1:].isdigit()]
            
            if measurement_regs:
                # Use the first available measurement register
                bitarray = getattr(circuit_result.data, measurement_regs[0])
            
            if bitarray is None:
                print(f"Available data attributes: {data_attrs}")
                raise ValueError("No measurement registers found in result")
                
            counts = bitarray_to_counts(bitarray)
            
        elif hasattr(circuit_result, 'quasi_dists'):
            # Handle quasi-distribution format
            counts = {format(k, f'0{len(qubit_line)}b'): int(v * circuit_result.shots) 
                     for k, v in circuit_result.quasi_dists[0].items()}
        else:
            print(f"Available result attributes: {dir(circuit_result)}")
            raise ValueError("Unexpected result format - no data or quasi_dists found")
        
        # Analysis logic remains the same
        if test_type == 'bell':
            remapped_counts = remap_counts(counts, qubit_line[:2])
            total = sum(remapped_counts.values())
            correct_states = ['00', '11']
            fidelity = sum(remapped_counts.get(state, 0) for state in correct_states) / total
            analyzed_data.append(fidelity)
        
        elif test_type == 'phase':
            remapped_counts = remap_counts(counts, qubit_line[:1])
            total = sum(remapped_counts.values())
            coherence = remapped_counts.get('0', 0) / total
            analyzed_data.append(coherence)
        
        elif test_type == 'ghz':
            remapped_counts = remap_counts(counts, qubit_line)
            total = sum(remapped_counts.values())
            expected_states = ['0' * len(qubit_line), '1' * len(qubit_line)]
            fidelity = sum(remapped_counts.get(state, 0) for state in expected_states) / total
            analyzed_data.append(fidelity)
    
    return analyzed_data

def print_experiment_summary(name, analyzed_data):
    """Print a summary of experimental results"""
    print(f"\n{name} Test Summary:")
    print("-" * 50)
    
    if not analyzed_data:
        print("No valid data")
        return
        
    print(f"Initial value: {analyzed_data[0]:.4f}")
    print(f"Final value: {analyzed_data[-1]:.4f}")
    print(f"Maximum: {max(analyzed_data):.4f}")
    print(f"Minimum: {min(analyzed_data):.4f}")
    print(f"Mean: {np.mean(analyzed_data):.4f}")
    print(f"Std dev: {np.std(analyzed_data):.4f}")

def plot_quantum_results(results_data, params, timestamp):
    """Plot quantum experiment results"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    delays = params['delays']
    
    # Plot settings for each test type with theoretical limits
    plot_configs = {
        'bell': {
            'title': 'Bell State Analysis',
            'metrics': {'Fidelity': 'analyzed_data'},
            'limits': {
                'Classical Limit': 0.7071,  # 1/√2
                'Heisenberg Limit': 1.0
            }
        },
        'phase': {
            'title': 'Phase Coherence Analysis',
            'metrics': {'Coherence': 'analyzed_data'},
            'limits': {
                'Random Phase': 0.5,
                'Perfect Coherence': 1.0
            }
        },
        'ghz': {
            'title': 'GHZ State Analysis',
            'metrics': {'Fidelity': 'analyzed_data'},
            'limits': {
                'Classical Bound': 0.5,
                'Genuine Entanglement': 0.5,  # Threshold for genuine n-partite entanglement
                'Perfect GHZ': 1.0
            }
        }
    }
    
    # Create plots
    for idx, (test_type, config) in enumerate(plot_configs.items()):
        if test_type in results_data:
            ax = axes[idx]
            test_data = results_data[test_type]
            
            # Plot metrics
            for label, key in config['metrics'].items():
                values = test_data[key]
                d = np.array(delays)/1000  # Convert to μs
                v = np.array(values)
                
                # Add small offset to zero values for log scale
                if np.any(v == 0):
                    min_nonzero = np.min(v[v > 0]) if np.any(v > 0) else 1e-10
                    v[v == 0] = min_nonzero/10
                
                ax.plot(d, v, 'o-', label=label)
            
            # Add theoretical limits
            colors = ['r', 'g', 'b']
            styles = ['--', ':', '-.']
            for (limit_name, limit_value), color, style in zip(config['limits'].items(), colors, styles):
                ax.axhline(y=limit_value, color=color, linestyle=style, 
                          label=limit_name, alpha=0.5)
            
            # Fill regions if applicable
            if test_type == 'bell':
                ax.fill_between(ax.get_xlim(), 0, 0.7071, color='red', alpha=0.1, 
                              label='Classical Region')
            elif test_type == 'ghz':
                ax.fill_between(ax.get_xlim(), 0, 0.5, color='red', alpha=0.1,
                              label='Classical Region')
            
            ax.set_xscale('log')
            ax.set_xlabel('Delay (μs)')
            ax.set_ylabel('Value')
            ax.set_title(config['title'])
            ax.set_ylim(0, 1.05)  # Set y-axis limits from 0 to just above 1
            
            # Add parameter text box
            param_text = (
                f"Qubit Line: {params['qubit_line']}\n"
                f"Protection Interval: {params['protection_interval']/1000:.0f}μs\n"
                f"Shots: 4096"
            )
            ax.text(0.02, 0.98, param_text,
                   transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                   verticalalignment='top',
                   fontsize=8)
            
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/results_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Initialize quantum service
        config = load_quantum_config()
        service = QiskitRuntimeService(
            channel=config['channel'],
            instance=config['instance']
        )
        
        # Get job information
        job_info = get_latest_job_info()
        timestamp = job_info['timestamp']
        qubit_line = job_info['parameters']['qubit_line']
        circuit_indices = job_info['circuit_indices']
        
        # Retrieve results from single job
        job = service.job(job_info['job_id'])
        all_results = job.result()
        
        # Analyze results for each experiment type
        results = {}
        for test_type in circuit_indices.keys():
            analyzed_data = analyze_quantum_results(
                all_results, 
                qubit_line, 
                test_type, 
                circuit_indices
            )
            results[test_type] = {
                'analyzed_data': analyzed_data
            }
            # Print summary for each test
            print_experiment_summary(test_type.upper(), analyzed_data)
        
        # Generate analysis timestamp in same format as simulate.py
        analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save analysis results with more metadata
        analysis_output = {
            'timestamp': analysis_timestamp,  # Use new timestamp format
            'backend': job_info['backend_name'],
            'parameters': {
                'qubit_line': qubit_line,
                'delays': job_info['parameters']['delays'],
                'protection_interval': job_info['parameters']['protection_interval'],
                'protection_angle': job_info['parameters']['protection_angle']
            },
            'data': results
        }
        
        os.makedirs('results', exist_ok=True)
        output_file = f'results/results_{analysis_timestamp}.json'  # Use new timestamp
        with open(output_file, 'w') as f:
            json.dump(analysis_output, f, indent=2)
        
        # Create plots with same timestamp
        plot_quantum_results(results, job_info['parameters'], analysis_timestamp)
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to:")
        print(f"- Data: {output_file}")
        print(f"- Plot: results/results_{analysis_timestamp}.png")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
