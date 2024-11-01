from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import TimeUnitConversion, ALAPSchedule
from datetime import datetime
import json
import os
import numpy as np
from dotenv import load_dotenv

from experiment import (
    calculate_delays,
    create_protection_circuits,
    create_bell_test_circuits,
    create_phase_test_circuits,
    create_ghz_test_circuits
)

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

def find_consecutive_line(coupling_map, min_length=7):
    """Find a consecutive line of connected qubits in the coupling map"""
    adj_list = {}
    for edge in coupling_map:
        q1, q2 = edge
        if q1 not in adj_list: adj_list[q1] = set()
        if q2 not in adj_list: adj_list[q2] = set()
        adj_list[q1].add(q2)
        adj_list[q2].add(q1)
    
    def dfs(start, visited, path):
        if len(path) >= min_length: return path
        visited.add(start)
        path.append(start)
        for next_qubit in adj_list[start]:
            if next_qubit not in visited:
                result = dfs(next_qubit, visited.copy(), path.copy())
                if result and len(result) >= min_length:
                    return result
        return None
    
    for start_qubit in adj_list:
        path = dfs(start_qubit, set(), [])
        if path and len(path) >= min_length:
            return path[:min_length]
    return None

def save_job_info(job_info):
    """Save job information with proper versioning"""
    os.makedirs('jobs', exist_ok=True)
    latest_path = 'jobs/latest.json'
    
    # If latest.json exists, archive it with its timestamp
    if os.path.exists(latest_path):
        with open(latest_path, 'r') as f:
            old_info = json.load(f)
            old_timestamp = old_info['timestamp'].replace(':', '-')
        archived_path = f'jobs/jobs_{old_timestamp}.json'
        os.rename(latest_path, archived_path)
    
    # Save new job info as latest.json
    with open(latest_path, 'w') as f:
        json.dump(job_info, f, indent=2)
    
    return latest_path

def main():
    try:
        # Initialize quantum service and get backend (unchanged)
        config = load_quantum_config()
        service = QiskitRuntimeService(
            token=config['token'],
            channel=config['channel'],
            instance=config['instance']
        )
        
        backend = service.least_busy(operational=True, simulator=False)
        print(f"Using backend: {backend.name}")
        print(f"Number of qubits: {backend.num_qubits}")
        
        # Find qubit line and calculate delays (unchanged)
        coupling_map = backend.configuration().coupling_map
        qubit_line = find_consecutive_line(coupling_map, min_length=7)
        if not qubit_line:
            raise ValueError("Could not find suitable qubit line")
        print(f"Using qubit line: {qubit_line}")
        
        delays = calculate_delays()
        print("\nUsing delays (ns):")
        for delay in delays:
            print(f"{delay:,} ns ({delay/1e6:.3f} ms)")
        
        # Create and collect all circuits
        circuit_sets = {
            'protection': create_protection_circuits(qubit_line, delays),
            'bell': create_bell_test_circuits(qubit_line, delays),
            'phase': create_phase_test_circuits(qubit_line, delays),
            'ghz': create_ghz_test_circuits(qubit_line, delays)
        }
        
        # Create optimization passes
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
        
        # Track circuit indices for each experiment
        circuit_indices = {}
        all_circuits = []
        current_index = 0
        
        # Combine and optimize all circuits
        for name, circuits in circuit_sets.items():
            optimized = scheduling_pm.run(init_pm.run(circuits))
            start_idx = current_index
            end_idx = current_index + len(optimized)
            circuit_indices[name] = (start_idx, end_idx)
            all_circuits.extend(optimized)
            current_index = end_idx
        
        # Submit single job
        print("\nSubmitting combined job...")
        sampler = Sampler(backend)
        job = sampler.run(all_circuits)
        print(f"Submitted job with {len(all_circuits)} circuits")
        
        # Save job info
        job_info = {
            'timestamp': datetime.now().isoformat(),
            'backend_name': backend.name,
            'backend_config': {
                'num_qubits': backend.num_qubits,
                'basis_gates': [str(gate) for gate in backend.basis_gates]
            },
            'job_id': job.job_id(),
            'circuit_indices': circuit_indices,
            'parameters': {
                'n_qubits': len(qubit_line),
                'qubit_line': qubit_line,
                'protection_angle': float(np.pi/4),
                'protection_interval': 50000,  # 0.05 ms
                'delays': delays
            }
        }
        
        # Save job info with versioning
        saved_path = save_job_info(job_info)
        
        print("\nJob submitted successfully!")
        print(f"Job info saved to: {saved_path}")
        print(f"Job ID: {job_info['job_id']}")
            
    except Exception as e:
        print(f"Error in job submission: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()