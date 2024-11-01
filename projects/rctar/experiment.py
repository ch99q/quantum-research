from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

def calculate_delays():
    """Calculate delay times that are divisible by 8"""
    desired_times = [
        0, # Initial rotation, to improve quantum state instantaneously. (Theoretical)
        16, 48, 96,  # Initial short delays
        1536,      # ~0.0015 ms
        24576,      # ~0.024 ms
        88888,      # ~0.09 ms
        100000,     # 0.1 ms
        200000,     # 0.2 ms
        1000000,    # 1.0 ms
        3000000,    # 3.0 ms
        5000000,    # 5.0 ms
        10000000    # 10.0 ms
    ]
    
    # Adjust times to be divisible by 8
    delays = [time + (8 - (time % 8)) if time % 8 != 0 else time for time in desired_times]

    print("\nUsing delays (ns):")
    return delays

def add_protection_sequence(circuit, qubit, duration, protection_interval=50000):
    """Add protection rotations at regular intervals"""
    angle = np.pi/4
    num_sequences = int(duration / protection_interval)
    
    for _ in range(num_sequences):
        circuit.delay(protection_interval, qubit)
        circuit.rz(angle, qubit)
        circuit.rx(angle/2, qubit)
    
    remaining_time = duration - (num_sequences * protection_interval)
    if remaining_time > 0:
        circuit.delay(remaining_time, qubit)

def create_base_circuit(qubit_line):
    """Create the base entangled state circuit"""
    n_qubits = len(qubit_line)
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qr, cr)

    # Create entangled state properly mapped to physical qubits
    for i in enumerate(qubit_line):
        qc.h(qr[i])  # Use logical index directly
    
    for i in range(n_qubits-1):
        qc.cx(qr[i], qr[i+1])  # Use sequential logical indices
    
    return qc, qr, cr

def create_protection_circuits(qubit_line, delays):
    """Create circuits for quantum protection validation"""
    circuits = []
    
    # Base circuit (no protection)
    qc_base, qr, cr = create_base_circuit(qubit_line)
    qc_base.barrier()
    qc_base.measure(qr, cr)
    circuits.append(qc_base)
    
    # Protected circuits with various delays
    for delay in delays:
        qc_protected = QuantumCircuit(qr, cr)
        
        # Initial state preparation
        for physical_qubit in qubit_line:
            qc_protected.h(qr[qubit_line.index(physical_qubit)])
        for i in range(len(qubit_line)-1):
            control_idx = qubit_line.index(qubit_line[i])
            target_idx = qubit_line.index(qubit_line[i+1])
            qc_protected.cx(qr[control_idx], qr[target_idx])
        
        qc_protected.barrier()
        
        # Add protection sequences for each qubit
        for physical_qubit in qubit_line:
            idx = qubit_line.index(physical_qubit)
            add_protection_sequence(qc_protected, qr[idx], delay)
        
        qc_protected.barrier()
        qc_protected.measure(qr, cr)
        circuits.append(qc_protected)
    
    return circuits

def create_bell_test_circuits(qubit_line, delays):
    """Create separate circuits for each delay in Bell state test"""
    circuits = []
    
    for delay in delays:
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        
        # Create Bell state
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.barrier()
        
        # Add protection sequences
        add_protection_sequence(qc, qr[0], delay)
        add_protection_sequence(qc, qr[1], delay)
        
        qc.barrier()
        qc.h(qr[0])  # Bell measurement
        qc.measure(qr, cr)
        
        circuits.append(qc)
    
    return circuits

def create_phase_test_circuits(qubit_line, delays):
    """Create separate circuits for each delay in phase test"""
    circuits = []
    
    for delay in delays:
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        
        # Create superposition with phase
        qc.h(qr[0])
        qc.rz(np.pi/4, qr[0])
        qc.barrier()
        
        # Add protection sequence
        add_protection_sequence(qc, qr[0], delay)
        
        qc.barrier()
        qc.rz(-np.pi/4, qr[0])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        
        circuits.append(qc)
    
    return circuits

def create_ghz_test_circuits(qubit_line, delays):
    """Create separate circuits for each delay in GHZ test"""
    circuits = []
    n_qubits = len(qubit_line)
    
    for delay in delays:
        qr = QuantumRegister(n_qubits)
        cr = ClassicalRegister(n_qubits)
        qc = QuantumCircuit(qr, cr)
        
        # Create GHZ state
        qc.h(qr[0])
        for j in range(n_qubits-1):
            qc.cx(qr[j], qr[j+1])
        qc.barrier()
        
        # Add protection sequences
        for j in range(n_qubits):
            add_protection_sequence(qc, qr[j], delay)
        
        qc.barrier()
        qc.h(qr)  # Apply H to all qubits
        qc.measure(qr, cr)
        
        circuits.append(qc)
    
    return circuits