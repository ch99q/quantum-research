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
    
    if any(t < 0 for t in desired_times):
        raise ValueError("Delays must be non-negative")
    
    # Adjust times to be divisible by 8
    delays = [time + (8 - (time % 8)) if time % 8 != 0 else time for time in desired_times]

    print("\nUsing delays (ns):")
    return delays

def add_protection_sequence(circuit, qubit, duration, protection_interval=50000):
    """Add protection rotations at regular intervals
    
    Args:
        circuit (QuantumCircuit): Circuit to add protection to
        qubit (Qubit): Target qubit
        duration (int): Total delay duration in ns
        protection_interval (int): Interval between protections in ns
    """
    if duration < 0 or protection_interval <= 0:
        raise ValueError("Duration and protection interval must be positive")
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
    for i in range(n_qubits):
        qc.h(qr[i]) # Use direct indices
    
    for i in range(n_qubits-1):
        qc.cx(qr[i], qr[i+1])  # Use sequential logical indices
    
    return qc, qr, cr

def create_protection_circuits(qubit_line, delays):
    """Create circuits for quantum protection validation"""
    n_qubits = len(qubit_line)
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    circuits = []
    
    # Base circuit (no protection)
    qc_base, qr, cr = create_base_circuit(qubit_line)
    qc_base.barrier()
    qc_base.measure(qr, cr)
    circuits.append(qc_base)
    
    # Protected circuits with various delays
    for delay in delays:
        qc_protected = QuantumCircuit(qr, cr)
        
        # Initial state preparation - use direct indices
        for i in range(n_qubits):
            qc_protected.h(qr[i])
        for i in range(n_qubits-1):
            qc_protected.cx(qr[i], qr[i+1])
        
        qc_protected.barrier()
        
        # Add protection sequences
        for i in range(n_qubits):
            add_protection_sequence(qc_protected, qr[i], delay)
        
        qc_protected.barrier()
        qc_protected.measure(qr, cr)
        circuits.append(qc_protected)
    
    return circuits

def create_bell_test_circuits(qubit_line, delays):
    """Create Bell state test circuits
    
    Args:
        qubit_line (list[int]): Physical qubit indices
        delays (list[int]): Delay times in nanoseconds
        
    Returns:
        list[QuantumCircuit]: List of test circuits
    """
    circuits = []
    
    for delay in delays:
        qr = QuantumRegister(len(qubit_line))
        cr = ClassicalRegister(len(qubit_line))
        qc = QuantumCircuit(qr, cr)
        
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.barrier()
        
        add_protection_sequence(qc, qr[0], delay)
        add_protection_sequence(qc, qr[1], delay)
        
        qc.barrier()
        qc.cx(qr[0], qr[1])  # Bell basis transformation
        qc.h(qr[0])
        qc.measure(qr[0:2], cr[0:2])
        
        circuits.append(qc)
    
    return circuits

def create_phase_test_circuits(qubit_line, delays):
    """Create phase coherence test circuits
    
    Args:
        qubit_line (list[int]): Physical qubit indices
        delays (list[int]): Delay times in nanoseconds
        
    Returns:
        list[QuantumCircuit]: List of test circuits
    """
    circuits = []
    
    for delay in delays:
        qr = QuantumRegister(len(qubit_line))
        cr = ClassicalRegister(len(qubit_line))
        qc = QuantumCircuit(qr, cr)
        
        qc.h(qr[0])
        qc.rz(np.pi/4, qr[0])
        qc.barrier()
        
        add_protection_sequence(qc, qr[0], delay)
        
        qc.barrier()
        qc.s(qr[0])  # S gate for π/2 phase
        qc.h(qr[0])  # Hadamard for phase measurement
        qc.measure(qr[0], cr[0])
        
        circuits.append(qc)
    
    return circuits

def create_ghz_test_circuits(qubit_line, delays):
    """Create GHZ state test circuits
    
    Args:
        qubit_line (list[int]): Physical qubit indices
        delays (list[int]): Delay times in nanoseconds
        
    Returns:
        list[QuantumCircuit]: List of test circuits
    """
    circuits = []
    n_qubits = len(qubit_line)
    
    for delay in delays:
        qr = QuantumRegister(n_qubits)
        cr = ClassicalRegister(n_qubits)
        qc = QuantumCircuit(qr, cr)
        
        qc.h(qr[0])
        for i in range(n_qubits-1):
            qc.cx(qr[i], qr[i+1])
        qc.barrier()
        
        for i in range(n_qubits):
            add_protection_sequence(qc, qr[i], delay)
        
        qc.barrier()
        for i in range(n_qubits-1):
            qc.cx(qr[i], qr[i+1])
        qc.h(qr[0])  # Reference basis
        qc.measure(qr, cr)
        
        circuits.append(qc)
    
    return circuits