from qiskit_ibm_runtime import QiskitRuntimeService
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import os
import time

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

def get_counts_from_result(result):
    """Extract counts from result"""
    if 'SamplerPubResult' in str(type(result)):
        data = result.data
        
        bitarray = None
        if hasattr(data, 'c0'):
            bitarray = data.c0
        elif hasattr(data, 'c1'):
            bitarray = data.c1
        
        if bitarray is not None:
            counts = {}
            try:
                raw_array = bitarray._array
                for shot_idx in range(bitarray.num_shots):
                    shot_value = raw_array[shot_idx][0]
                    key = format(shot_value, f'0{bitarray.num_bits}b')
                    counts[key] = counts.get(key, 0) + 1
                return counts
            except AttributeError:
                for shot_idx in range(bitarray.num_shots):
                    shot = []
                    for bit_idx in range(bitarray.num_bits):
                        val = bitarray[shot_idx, bit_idx]
                        shot.append(str(int(val)))
                    bitstring = ''.join(shot)
                    counts[bitstring] = counts.get(bitstring, 0) + 1
                return counts

    raise ValueError(f"Unsupported result format: {type(result)}")

def remap_counts(counts, qubit_line):
    """Remap counts from physical to logical qubits"""
    physical_to_logical = {phys: log for log, phys in enumerate(qubit_line)}
    remapped_counts = {}
    
    for bitstring, count in counts.items():
        bitstring = bitstring.zfill(len(qubit_line))
        remapped_bits = ['0'] * len(qubit_line)
        
        for phys_pos, bit in enumerate(bitstring):
            if phys_pos in physical_to_logical:
                logical_pos = physical_to_logical[phys_pos]
                if logical_pos < len(remapped_bits):
                    remapped_bits[logical_pos] = bit
        
        remapped_key = ''.join(remapped_bits)
        remapped_counts[remapped_key] = remapped_counts.get(remapped_key, 0) + count
    
    return remapped_counts

def analyze_bell_state(counts):
    """Analyze Bell state measurement results"""
    total = sum(counts.values())
    correct_states = ['00', '11']  # Expected Bell state outcomes
    fidelity = sum(counts.get(state, 0) for state in correct_states) / total
    return fidelity

def analyze_phase_coherence(counts):
    """Analyze phase coherence measurement results"""
    total = sum(counts.values())
    # For perfect phase coherence, expect '0' state
    coherence = counts.get('0', 0) / total
    return coherence

def analyze_ghz_state(counts, n_qubits):
    """Analyze GHZ state measurement results"""
    total = sum(counts.values())
    # GHZ state should give equal superposition of all 0s and all 1s
    expected_states = ['0' * n_qubits, '1' * n_qubits]
    fidelity = sum(counts.get(state, 0) for state in expected_states) / total
    return fidelity

def calculate_protection_strength(counts, qubit_line):
    """Calculate protection strength and uncertainty"""
    remapped_counts = remap_counts(counts, qubit_line)
    total = sum(remapped_counts.values())
    
    if total == 0:
        return 0.0, 0.0
    
    expected = '0' * len(qubit_line)
    success_counts = remapped_counts.get(expected, 0)
    fidelity = success_counts / total
    uncertainty = np.sqrt((fidelity * (1 - fidelity)) / total)
    
    return fidelity, uncertainty

def plot_quantum_analysis(delays, protection_strengths, bell_fidelities, 
                         phase_coherences, ghz_fidelities, timestamp=None):
    """Plot comprehensive quantum analysis results"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(15, 12))
    
    # Protection Strength vs Classical Threshold
    plt.subplot(221)
    # Skip the first two results (base and protected circuits)
    plt.plot(delays, protection_strengths[2:], 'bo-', label='Protection Strength')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Classical Threshold')
    plt.xscale('log')
    plt.xlabel('Delay (ns)')
    plt.ylabel('Strength')
    plt.title('Protection Strength vs Classical Threshold')
    plt.grid(True)
    plt.legend()
    
    # Bell State Fidelity
    plt.subplot(222)
    plt.plot(delays, bell_fidelities, 'go-', label='Bell State Fidelity')
    plt.axhline(y=0.7071, color='r', linestyle='--', label='Classical Limit')
    plt.xscale('log')
    plt.xlabel('Delay (ns)')
    plt.ylabel('Fidelity')
    plt.title('Bell State Fidelity Over Time')
    plt.grid(True)
    plt.legend()
    
    # Phase Coherence
    plt.subplot(223)
    plt.plot(delays, phase_coherences, 'mo-', label='Phase Coherence')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Phase')
    plt.xscale('log')
    plt.xlabel('Delay (ns)')
    plt.ylabel('Coherence')
    plt.title('Phase Coherence Over Time')
    plt.grid(True)
    plt.legend()
    
    # GHZ State Fidelity
    plt.subplot(224)
    plt.plot(delays, ghz_fidelities, 'co-', label='GHZ Fidelity')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Classical Limit')
    plt.xscale('log')
    plt.xlabel('Delay (ns)')
    plt.ylabel('Fidelity')
    plt.title('GHZ State Fidelity Over Time')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/quantum_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Load job information
        with open('jobs.json', 'r') as f:
            job_info = json.load(f)
        
        # Initialize quantum service
        config = load_quantum_config()
        service = QiskitRuntimeService(
            channel=config['channel'],
            instance=config['instance']
        )
        
        # Get results
        protection_results = service.job(job_info['jobs']['protection']).result()
        validation_results = service.job(job_info['jobs']['validation']).result()
        
        qubit_line = job_info['parameters']['qubit_line']
        delays = job_info['parameters']['delays']
        n_qubits = job_info['parameters']['n_qubits']
        
        # Analyze protection results
        protection_strengths = []
        protection_uncertainties = []
        
        for result in protection_results:
            counts = get_counts_from_result(result)
            strength, uncertainty = calculate_protection_strength(counts, qubit_line)
            protection_strengths.append(strength)
            protection_uncertainties.append(uncertainty)
        
        # Store base and protected circuit results
        base_strength = protection_strengths[0]
        protected_strength = protection_strengths[1]
        
        # Analyze quantum validation results
        bell_fidelities = []
        phase_coherences = []
        ghz_fidelities = []
        
        for i in range(0, len(validation_results), 3):  # Process results in groups of 3
            # Bell state results
            bell_counts = get_counts_from_result(validation_results[i])
            bell_fidelity = analyze_bell_state(bell_counts)
            bell_fidelities.append(bell_fidelity)
            
            # Phase coherence results
            phase_counts = get_counts_from_result(validation_results[i+1])
            coherence = analyze_phase_coherence(phase_counts)
            phase_coherences.append(coherence)
            
            # GHZ state results
            ghz_counts = get_counts_from_result(validation_results[i+2])
            ghz_fidelity = analyze_ghz_state(ghz_counts, n_qubits)
            ghz_fidelities.append(ghz_fidelity)
        
        # Plot results with delay-based results only
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_quantum_analysis(delays, protection_strengths, bell_fidelities,
                            phase_coherences, ghz_fidelities, timestamp)
        
        # Save analysis results
        analysis_results = {
            'timestamp': timestamp,
            'protection': {
                'base_strength': float(base_strength),
                'protected_strength': float(protected_strength),
                'delays': delays,
                'delay_strengths': protection_strengths[2:],  # Skip base and protected
                'uncertainties': protection_uncertainties[2:],  # Skip base and protected
                'mean_strength': float(np.mean(protection_strengths[2:])),
                'max_strength': float(max(protection_strengths[2:])),
                'max_strength_delay': delays[protection_strengths[2:].index(max(protection_strengths[2:]))]
            },
            'quantum_validation': {
                'bell_fidelities': bell_fidelities,
                'phase_coherences': phase_coherences,
                'ghz_fidelities': ghz_fidelities,
                'mean_bell_fidelity': float(np.mean(bell_fidelities)),
                'mean_phase_coherence': float(np.mean(phase_coherences)),
                'mean_ghz_fidelity': float(np.mean(ghz_fidelities))
            }
        }
        
        os.makedirs('results', exist_ok=True)
        with open(f'results/quantum_analysis_{timestamp}.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print("\nAnalysis Results:")
        print(f"Protection:")
        print(f"  Base Strength: {base_strength:.4f}")
        print(f"  Protected Strength: {protected_strength:.4f}")
        print(f"  Mean Delay Strength: {np.mean(protection_strengths[2:]):.4f} ± {np.std(protection_strengths[2:]):.4f}")
        print(f"  Max Delay Strength: {max(protection_strengths[2:]):.4f} at {delays[protection_strengths[2:].index(max(protection_strengths[2:]))]}ns")
        print(f"\nQuantum Validation:")
        print(f"  Mean Bell State Fidelity: {np.mean(bell_fidelities):.4f}")
        print(f"  Mean Phase Coherence: {np.mean(phase_coherences):.4f}")
        print(f"  Mean GHZ State Fidelity: {np.mean(ghz_fidelities):.4f}")
        print(f"\nQuantum Nature Assessment:")
        print(f"  Bell State Test: {'Quantum' if np.mean(bell_fidelities) > 0.7071 else 'Classical'}")
        print(f"  Phase Coherence: {'Maintained' if np.mean(phase_coherences) > 0.5 else 'Lost'}")
        print(f"  GHZ State Test: {'Quantum' if np.mean(ghz_fidelities) > 0.5 else 'Classical'}")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()