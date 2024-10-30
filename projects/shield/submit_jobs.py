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

def create_protection_circuits(qubit_line):
    """Create circuits for quantum protection validation"""
    n_qubits = len(qubit_line)
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    circuits = []
    
    # Base circuit
    qc_base = QuantumCircuit(qr, cr)
    for physical_qubit in qubit_line:
        qc_base.h(qr[qubit_line.index(physical_qubit)])
    for i in range(n_qubits-1):
        control_idx = qubit_line.index(qubit_line[i])
        target_idx = qubit_line.index(qubit_line[i+1])
        qc_base.cx(qr[control_idx], qr[target_idx])
    qc_base.barrier()
    qc_base.measure(qr, cr)
    circuits.append(qc_base)
    
    # Protected circuit
    qc_protected = QuantumCircuit(qr, cr)
    for physical_qubit in qubit_line:
        qc_protected.h(qr[qubit_line.index(physical_qubit)])
    for i in range(n_qubits-1):
        control_idx = qubit_line.index(qubit_line[i])
        target_idx = qubit_line.index(qubit_line[i+1])
        qc_protected.cx(qr[control_idx], qr[target_idx])
    angle = np.pi/4
    for physical_qubit in qubit_line:
        idx = qubit_line.index(physical_qubit)
        qc_protected.rz(angle, qr[idx])
        qc_protected.rx(angle/2, qr[idx])
    qc_protected.barrier()
    qc_protected.measure(qr, cr)
    circuits.append(qc_protected)
    
    # Delay tests
    delays = [16, 48, 96, 192, 384, 768, 1536, 6144, 24576, 98304, 393216, 786432, 3145728, 12582912, 50331648]
    for delay in delays:
        qc_delay = QuantumCircuit(qr, cr)
        for physical_qubit in qubit_line:
            qc_delay.h(qr[qubit_line.index(physical_qubit)])
        for i in range(n_qubits-1):
            control_idx = qubit_line.index(qubit_line[i])
            target_idx = qubit_line.index(qubit_line[i+1])
            qc_delay.cx(qr[control_idx], qr[target_idx])
        for physical_qubit in qubit_line:
            idx = qubit_line.index(physical_qubit)
            qc_delay.rz(angle, qr[idx])
            qc_delay.rx(angle/2, qr[idx])
        qc_delay.barrier()
        for physical_qubit in qubit_line:
            qc_delay.delay(delay, qr[qubit_line.index(physical_qubit)])
        qc_delay.barrier()
        qc_delay.measure(qr, cr)
        circuits.append(qc_delay)
    
    return circuits

def create_quantum_validation_circuits(qubit_line, delays):
    """Create circuits to validate quantum behavior"""
    n_qubits = len(qubit_line)
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    circuits = []
    
    # Bell State Test
    for delay in delays:
        qc_bell = QuantumCircuit(qr, cr)
        # Create Bell state
        qc_bell.h(qr[0])
        qc_bell.cx(qr[0], qr[1])
        qc_bell.barrier()
        # Add delay
        qc_bell.delay(delay, qr[0])
        qc_bell.delay(delay, qr[1])
        qc_bell.barrier()
        # Bell measurement
        qc_bell.h(qr[0])
        qc_bell.measure(qr, cr)
        circuits.append(qc_bell)
    
    # Phase Coherence Test
    for delay in delays:
        qc_phase = QuantumCircuit(qr, cr)
        # Create superposition with phase
        qc_phase.h(qr[0])
        qc_phase.rz(np.pi/4, qr[0])
        qc_phase.barrier()
        # Add delay
        qc_phase.delay(delay, qr[0])
        qc_phase.barrier()
        # Reverse phase and measure
        qc_phase.rz(-np.pi/4, qr[0])
        qc_phase.h(qr[0])
        qc_phase.measure(qr, cr)
        circuits.append(qc_phase)
    
    # GHZ State Test
    for delay in delays:
        qc_ghz = QuantumCircuit(qr, cr)
        # Create GHZ state
        qc_ghz.h(qr[0])
        for i in range(n_qubits-1):
            qc_ghz.cx(qr[i], qr[i+1])
        qc_ghz.barrier()
        # Add delay
        for i in range(n_qubits):
            qc_ghz.delay(delay, qr[i])
        qc_ghz.barrier()
        # Measure in superposition basis
        for i in range(n_qubits):
            qc_ghz.h(qr[i])
        qc_ghz.measure(qr, cr)
        circuits.append(qc_ghz)
    
    return circuits

def main():
    try:
        # Initialize quantum service
        config = load_quantum_config()
        service = QiskitRuntimeService(
            token=config['token'],
            channel=config['channel'],
            instance=config['instance']
        )
        service.save_account(token=config['token'], channel=config['channel'], overwrite=True)
        
        # Get backend
        backend = service.least_busy(operational=True, simulator=False)
        print(f"Using backend: {backend.name}")
        print(f"Number of qubits: {backend.num_qubits}")
        
        # Find qubit line
        coupling_map = backend.configuration().coupling_map
        qubit_line = find_consecutive_line(coupling_map, min_length=7)
        if not qubit_line:
            raise ValueError("Could not find suitable qubit line")
        print(f"Using qubit line: {qubit_line}")
        
        # Create and optimize circuits
        delays = [16, 48, 96, 192, 384, 768, 1536, 6144, 24576, 98304, 393216, 786432, 3145728, 12582912, 50331648]
        protection_circuits = create_protection_circuits(qubit_line)
        quantum_validation_circuits = create_quantum_validation_circuits(qubit_line, delays)
        
        # Optimize circuits
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
        
        protection_optimized = scheduling_pm.run(init_pm.run(protection_circuits))
        validation_optimized = scheduling_pm.run(init_pm.run(quantum_validation_circuits))
        
        # Create sampler and submit jobs
        sampler = Sampler(backend)
        print("\nSubmitting jobs...")
        jobs = {
            'protection': sampler.run(protection_optimized),
            'validation': sampler.run(validation_optimized)
        }
        
        # Save job info
        job_info = {
            'timestamp': datetime.now().isoformat(),
            'backend_name': backend.name,
            'backend_config': {
                'num_qubits': backend.num_qubits,
                'basis_gates': [str(gate) for gate in backend.basis_gates]
            },
            'jobs': {
                'protection': jobs['protection'].job_id(),
                'validation': jobs['validation'].job_id()
            },
            'parameters': {
                'n_qubits': len(qubit_line),
                'qubit_line': qubit_line,
                'protection_angle': float(np.pi/4),
                'delays': delays
            }
        }
        
        with open('jobs.json', 'w') as f:
            json.dump(job_info, f, indent=2)
        
        print("\nJobs submitted successfully!")
        print("Job IDs saved to jobs.json")
        print("\nJob IDs:")
        for exp_name, job_id in job_info['jobs'].items():
            print(f"{exp_name}: {job_id}")
            
    except Exception as e:
        print(f"Error in job submission: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()