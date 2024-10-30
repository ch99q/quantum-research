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
    """Extract counts from result, handling different result formats including SamplerPubResult"""
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
    """Remap counts from physical to logical qubit ordering"""
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

def calculate_protection_strength(counts, qubit_line):
    """Calculate protection strength and its uncertainty from measurement results"""
    remapped_counts = remap_counts(counts, qubit_line)
    total = sum(remapped_counts.values())
    
    if total == 0:
        return 0.0, 0.0
    
    expected = '0' * len(qubit_line)
    success_counts = remapped_counts.get(expected, 0)
    fidelity = success_counts / total
    
    uncertainty = np.sqrt((fidelity * (1 - fidelity)) / total)
    
    return fidelity, uncertainty

def calculate_uncertainty_product(pos_counts, mom_counts, qubit_line):
    """Calculate uncertainty product from position and momentum measurements"""
    def calculate_variance_with_uncertainty(counts):
        remapped_counts = remap_counts(counts, qubit_line)
        total = sum(remapped_counts.values())
        
        mean = sum(sum(int(bit) * 2**i for i, bit in enumerate(key)) * count 
                  for key, count in remapped_counts.items()) / total
                  
        var = sum(sum((int(bit) * 2**i - mean)**2 for i, bit in enumerate(key)) * count 
                 for key, count in remapped_counts.items()) / total
                 
        var_of_var = (2 * var**2) / (total - 1)
        uncertainty = np.sqrt(var_of_var)
            
        return var, uncertainty
    
    var_x, unc_x = calculate_variance_with_uncertainty(pos_counts)
    var_p, unc_p = calculate_variance_with_uncertainty(mom_counts)
    
    dx = np.sqrt(var_x)
    dp = np.sqrt(var_p)
    
    dx_unc = unc_x / (2 * np.sqrt(var_x)) if var_x > 0 else 0
    dp_unc = unc_p / (2 * np.sqrt(var_p)) if var_p > 0 else 0
    
    product = dx * dp
    product_uncertainty = product * np.sqrt((dx_unc/dx)**2 + (dp_unc/dp)**2) if product > 0 else 0
    
    return product, product_uncertainty

def plot_protection_results(delays, strengths, uncertainties, timestamp=None):
    """Plot protection measurement results"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(211)
    plt.errorbar(delays, strengths, yerr=uncertainties, 
                fmt='bo-', capsize=5, markersize=8, 
                label='Protection Strength')
    
    plt.fill_between(delays, 
                    [s-u for s,u in zip(strengths, uncertainties)],
                    [s+u for s,u in zip(strengths, uncertainties)],
                    alpha=0.2, color='blue')
    
    plt.axhline(y=0.5, color='r', linestyle='--', label='Theoretical Maximum')
    plt.annotate('Theoretical Maximum', xy=(delays[0], 0.51), 
                xytext=(delays[0] + 10, 0.52),
                arrowprops=dict(facecolor='red', shrink=0.05))
    
    mean_strength = np.mean(strengths)
    plt.axhline(y=mean_strength, color='g', linestyle=':', 
                label=f'Mean = {mean_strength:.3f}')
    
    plt.xlabel('Delay (ns)')
    plt.ylabel('Protection Strength')
    plt.title('Quantum Protection vs Delay')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.ylim([0, max(max(strengths) + max(uncertainties), 0.55)])
    
    plt.subplot(212)
    
    q75, q25 = np.percentile(strengths, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(strengths) ** (1/3))
    n_bins = int(np.ceil((max(strengths) - min(strengths)) / bin_width)) if bin_width > 0 else 5
    
    plt.hist(strengths, bins=n_bins, density=True, alpha=0.7, 
            color='blue', edgecolor='black', label='Measurements')
    
    mean_strength = np.mean(strengths)
    std_strength = np.std(strengths)
    plt.axvline(mean_strength, color='r', linestyle='-', 
                label=f'Mean = {mean_strength:.3f}')
    plt.axvline(mean_strength + std_strength, color='g', linestyle='--', 
                label=f'±1σ = {std_strength:.3f}')
    plt.axvline(mean_strength - std_strength, color='g', linestyle='--')
    
    plt.xlabel('Protection Strength')
    plt.ylabel('Density')
    plt.title('Distribution of Protection Strengths')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/protection_results_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_validation_results(uncertainty_product, product_uncertainty, timestamp=None):
    """Plot uncertainty relation validation results"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    plt.figure(figsize=(10, 6))
    
    plt.bar(['Uncertainty Product'], [uncertainty_product], 
            yerr=[product_uncertainty], capsize=5,
            color='blue', alpha=0.7, label='Measured Value')
    
    plt.axhline(y=0.5, color='r', linestyle='--', 
                label='Heisenberg Limit (ℏ/2)')
    
    plt.fill_between(['Uncertainty Product'], 
                    [uncertainty_product - product_uncertainty], 
                    [uncertainty_product + product_uncertainty],
                    alpha=0.2, color='blue')
    
    plt.annotate(f'{uncertainty_product:.3f} ± {product_uncertainty:.3f}ℏ',
                xy=('Uncertainty Product', uncertainty_product),
                xytext=(0.2, uncertainty_product + 0.1),
                ha='center',
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.ylabel('Uncertainty Product (ℏ units)')
    plt.title('Heisenberg Uncertainty Relation Validation')
    
    satisfaction_text = "Heisenberg Limit Satisfied" if uncertainty_product > 0.5 else "Below Heisenberg Limit"
    satisfaction_color = "green" if uncertainty_product > 0.5 else "red"
    plt.text(0.5, -0.1, satisfaction_text,
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             color=satisfaction_color,
             fontsize=12,
             fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.ylim([0, max(uncertainty_product + product_uncertainty * 2, 0.7)])
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/uncertainty_validation_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        with open('jobs.json', 'r') as f:
            job_info = json.load(f)
        
        config = load_quantum_config()
        service = QiskitRuntimeService(
            channel=config['channel'],
            instance=config['instance']
        )
        
        protection_results = service.job(job_info['jobs']['protection']).result()
        validation_results = service.job(job_info['jobs']['validation']).result()
        
        qubit_line = job_info['parameters']['qubit_line']
        delays = [0] + job_info['parameters'].get('delays', [16, 48, 96, 192])
        
        strengths = []
        uncertainties = []
        
        for i, result in enumerate(protection_results):
            if i >= len(delays):
                break
                
            counts = get_counts_from_result(result)
            strength, uncertainty = calculate_protection_strength(counts, qubit_line)
            strengths.append(strength)
            uncertainties.append(uncertainty)
        
        pos_counts = get_counts_from_result(validation_results[0])
        mom_counts = get_counts_from_result(validation_results[1])
        
        uncertainty_product, product_uncertainty = calculate_uncertainty_product(
            pos_counts, mom_counts, qubit_line
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_data = {
            'timestamp': timestamp,
            'protection': {
                'delays': delays[:len(strengths)],
                'strengths': strengths,
                'uncertainties': uncertainties,
                'statistics': {
                    'mean_strength': float(np.mean(strengths)),
                    'std_strength': float(np.std(strengths)),
                    'max_strength': float(max(strengths)),
                    'max_strength_delay': delays[strengths.index(max(strengths))]
                }
            },
            'validation': {
                'uncertainty_product': float(uncertainty_product),
                'product_uncertainty': float(product_uncertainty),
                'heisenberg_satisfied': bool(uncertainty_product > 0.5)
            },
            'qubit_line': qubit_line
        }
        
        os.makedirs('results', exist_ok=True)
        with open(f'results/quantum_results_{timestamp}.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print("\nResults Summary:")
        print(f"Protection:")
        print(f"  Mean Strength: {np.mean(strengths):.4f} ± {np.std(strengths):.4f}")
        print(f"  Max Strength: {max(strengths):.4f} at {delays[strengths.index(max(strengths))]}ns")
        print(f"Validation:")
        print(f"  Uncertainty Product: {uncertainty_product:.4f} ± {product_uncertainty:.4f}ℏ")
        print(f"  Heisenberg Limit Satisfied: {uncertainty_product > 0.5}")
        
        plot_protection_results(delays[:len(strengths)], strengths, uncertainties, timestamp)
        plot_validation_results(uncertainty_product, product_uncertainty, timestamp)
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        raise

if __name__ == "__main__":
    main()