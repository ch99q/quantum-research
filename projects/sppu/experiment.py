from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import TimeUnitConversion, ALAPSchedule
import numpy as np
import json
from datetime import datetime
from dotenv import load_dotenv
import os

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

def create_XX_YY_coupling(qc, qubit1, qubit2):
    """Create XX+YY coupling between two qubits"""
    # Implement e^(-iπ/4(XX + YY))
    qc.h([qubit1, qubit2])
    qc.cx(qubit1, qubit2)
    qc.rz(np.pi/2, qubit2)
    qc.cx(qubit1, qubit2)
    qc.h([qubit1, qubit2])

def apply_protection_cycle(qc, qubits, angle):
    """Apply a single protection cycle"""
    # Protection rotations
    for q in qubits:
        qc.rz(angle, q)
        qc.sx(q)  # √X gate
        qc.rz(angle/2, q)
        qc.sx(q)
    
    # XX+YY coupling network
    for i in range(len(qubits)-1):
        create_XX_YY_coupling(qc, qubits[i], qubits[i+1])

def create_protected_circuit(qubit_line, delay, protection_interval=64, angle=np.pi/4):
    """Create protected quantum circuit with periodic protection"""
    n_qubits = len(qubit_line)
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    
    # Create circuits for different measurement bases
    circuits = []
    
    # Position measurement circuit
    qc_pos = QuantumCircuit(qr, cr)
    
    # Initial state preparation
    qc_pos.h(qr[0])
    qc_pos.x(qr[1])
    qc_pos.h(qr[1])
    
    # Apply initial protection
    apply_protection_cycle(qc_pos, range(n_qubits), angle)
    
    # Add delay with periodic protection
    remaining_delay = delay
    while remaining_delay > 0:
        current_delay = min(protection_interval, remaining_delay)
        
        # Add delay
        for q in range(n_qubits):
            qc_pos.delay(current_delay, q)
        
        remaining_delay -= current_delay
        
        if remaining_delay > 0:
            # Apply protection cycle
            apply_protection_cycle(qc_pos, range(n_qubits), angle)
    
    # Position measurement
    qc_pos.h(range(n_qubits))
    qc_pos.measure(qr, cr)
    circuits.append(qc_pos)
    
    # Momentum measurement circuit
    qc_mom = QuantumCircuit(qr, cr)
    # Copy the circuit up to measurement
    qc_mom.compose(qc_pos, inplace=True)
    # Remove measurements
    qc_mom.remove_final_measurements()
    
    # Momentum basis measurement
    for q in range(n_qubits):
        qc_mom.s(q)
        qc_mom.h(q)
    qc_mom.measure(qr, cr)
    circuits.append(qc_mom)
    
    return circuits

def optimize_circuits(circuits, backend):
    """Optimize circuits for the backend"""
    init_pm = generate_preset_pass_manager(
        optimization_level=1,
        basis_gates=backend.configuration().basis_gates,
        coupling_map=backend.configuration().coupling_map,
        target=backend.target
    )
    
    scheduling_pm = PassManager([
        TimeUnitConversion(target=backend.target),
        ALAPSchedule(backend.instruction_durations)
    ])
    
    optimized = init_pm.run(circuits)
    scheduled = scheduling_pm.run(optimized)
    
    return scheduled

def analyze_results(results, n_qubits, delays):
   """Analyze measurement results for protection validation"""
   protection_strengths = []
   uncertainties = []
   network_coherence = []
   
   for i, delay in enumerate(delays):
       # Get position and momentum measurements for this delay
       pos_counts = results[i*2].quasi_dists[0]
       mom_counts = results[i*2 + 1].quasi_dists[0]
       
       # Calculate expectation values
       pos_exp = calculate_expectation(pos_counts)
       mom_exp = calculate_expectation(mom_counts)
       
       # Calculate protection strength
       p_strength = (abs(pos_exp)**2 + abs(mom_exp)**2) / 2
       protection_strengths.append(p_strength)
       
       # Calculate uncertainty product
       pos_var = calculate_variance(pos_counts)
       mom_var = calculate_variance(mom_counts)
       uncertainty = np.sqrt(pos_var * mom_var)
       uncertainties.append(uncertainty)
       
       # Calculate network coherence
       coherence = calculate_network_coherence(pos_counts, mom_counts, n_qubits)
       network_coherence.append(coherence)
   
   return {
       'protection_strengths': protection_strengths,
       'uncertainties': uncertainties,
       'network_coherence': network_coherence
   }

def calculate_expectation(counts):
   """Calculate expectation value from measurement counts"""
   total = sum(counts.values())
   exp_val = sum((-1)**bin(state).count('1') * count / total 
                for state, count in counts.items())
   return exp_val

def calculate_variance(counts):
   """Calculate variance of measurements"""
   exp_val = calculate_expectation(counts)
   total = sum(counts.values())
   var = sum((-1)**bin(state).count('1')**2 * count / total 
             for state, count in counts.items()) - abs(exp_val)**2
   return var

def calculate_network_coherence(pos_counts, mom_counts, n_qubits):
   """Calculate network coherence from XX+YY correlations"""
   # Calculate adjacent qubit correlations
   coherence = 0
   for i in range(n_qubits - 1):
       xx_corr = calculate_correlation(pos_counts, i, i+1)
       yy_corr = calculate_correlation(mom_counts, i, i+1)
       coherence += np.sqrt(xx_corr**2 + yy_corr**2)
   return coherence / (n_qubits - 1)

def calculate_correlation(counts, i, j):
   """Calculate correlation between qubits i and j"""
   total = sum(counts.values())
   corr = 0
   for state, count in counts.items():
       bits = format(state, f'0{len(bin(state))-2}b')
       val = (-1)**(int(bits[i]) + int(bits[j]))
       corr += val * count / total
   return corr

def validate_results(experiment_data):
   """Validate experimental results against theoretical predictions"""
   results = experiment_data['results']
   params = experiment_data['parameters']
   
   # Calculate theoretical predictions
   theoretical_max = 0.5
   theoretical_uncertainty = 0.5 * np.pi  # ℏ/2
   
   # Validation metrics
   validation = {
       'protection_strength': {
           'mean': np.mean(results['protection_strengths']),
           'std': np.std(results['protection_strengths']),
           'max': np.max(results['protection_strengths']),
           'theoretical_ratio': np.mean(results['protection_strengths']) / theoretical_max
       },
       'uncertainty_product': {
           'mean': np.mean(results['uncertainties']),
           'std': np.std(results['uncertainties']),
           'min': np.min(results['uncertainties']),
           'heisenberg_satisfied': all(u >= theoretical_uncertainty 
                                    for u in results['uncertainties'])
       },
       'network_coherence': {
           'mean': np.mean(results['network_coherence']),
           'std': np.std(results['network_coherence']),
           'decay_rate': (results['network_coherence'][0] - 
                        results['network_coherence'][-1]) / params['delays'][-1]
       }
   }
   
   return validation

def submit_protection_experiment():
    """Submit the protection experiment to IBM"""
    config = load_quantum_config()
    service = QiskitRuntimeService(
        token=config['token'],
        channel=config['channel'],
        instance=config['instance']
    )
    
    # Get Eagle R3 backend
    backend = service.get_backend('ibm_eagle_r3')
    print(f"Using backend: {backend.name}")
    
    # Define experimental parameters
    delays = [16, 48, 96, 192, 384, 768, 1536, 6144, 24576, 98304]
    protection_interval = 64  # ns
    protection_angle = np.pi/4
    qubit_line = [1, 0, 14, 18, 19, 20, 33]  # Eagle R3 specific
    
    # Create and optimize circuits
    all_circuits = []
    for delay in delays:
        circuits = create_protected_circuit(
            qubit_line, 
            delay,
            protection_interval,
            protection_angle
        )
        all_circuits.extend(circuits)
    
    optimized_circuits = optimize_circuits(all_circuits, backend)
    
    # Submit jobs
    sampler = Sampler(
        backend,
        options={
            "shots": 4000,
            "optimization_level": 1,
            "resilience_level": 1
        }
    )
    
    job = sampler.run(optimized_circuits)
    
    # Save job information
    job_info = {
        'timestamp': datetime.now().isoformat(),
        'backend_name': backend.name,
        'job_id': job.job_id(),
        'parameters': {
            'delays': delays,
            'protection_interval': protection_interval,
            'protection_angle': float(protection_angle),
            'qubit_line': qubit_line
        },
        'status': 'submitted'
    }
    
    with open('protection_jobs.json', 'w') as f:
        json.dump(job_info, f, indent=2)
    
    print(f"\nExperiment submitted successfully!")
    print(f"Job ID: {job.job_id()}")
    print("Job information saved to protection_jobs.json")

def check_and_analyze_results():
    """Check job status and analyze results if complete"""
    if not os.path.exists('protection_jobs.json'):
        print("No job information found. Please submit the experiment first.")
        return
    
    try:
        # Load job information
        with open('protection_jobs.json', 'r') as f:
            job_info = json.load(f)
        
        if job_info['status'] == 'completed':
            print("Results already analyzed.")
            return
        
        # Connect to IBM
        config = load_quantum_config()
        service = QiskitRuntimeService(
            token=config['token'],
            channel=config['channel'],
            instance=config['instance']
        )
        
        # Get job
        job = service.job(job_info['job_id'])
        
        # Check status
        status = job.status()
        print(f"Job status: {status}")
        
        if status != 'COMPLETED':
            print("Job not yet completed. Please try again later.")
            return
        
        # Get results
        results = job.result()
        
        # Analyze results
        analysis = analyze_results(
            results,
            len(job_info['parameters']['qubit_line']),
            job_info['parameters']['delays']
        )
        
        # Update job info with results
        job_info['status'] = 'completed'
        job_info['results'] = {
            'protection_strengths': [float(x) for x in analysis['protection_strengths']],
            'uncertainties': [float(x) for x in analysis['uncertainties']],
            'network_coherence': [float(x) for x in analysis['network_coherence']]
        }
        
        # Save updated information
        with open('protection_jobs.json', 'w') as f:
            json.dump(job_info, f, indent=2)
        
        # Save detailed results
        results_filename = f"protection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_filename, 'w') as f:
            json.dump(job_info, f, indent=2)
        
        # Validate results
        validation = validate_results(job_info)
        
        # Print summary
        print("\nResults Analysis Complete!")
        print("\nExperiment Summary:")
        print(f"Protection Strength: {validation['protection_strength']['mean']:.3f} ± "
              f"{validation['protection_strength']['std']:.3f}")
        print(f"Uncertainty Product: {validation['uncertainty_product']['mean']:.3f} ± "
              f"{validation['uncertainty_product']['std']:.3f}")
        print(f"Network Coherence: {validation['network_coherence']['mean']:.3f} ± "
              f"{validation['network_coherence']['std']:.3f}")
        
        print(f"\nDetailed results saved to {results_filename}")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if job information exists
    if not os.path.exists('protection_jobs.json'):
        print("Submitting new experiment...")
        submit_protection_experiment()
    else:
        print("Checking results...")
        check_and_analyze_results()