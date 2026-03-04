# Quantum Optimization Search for H(668)

## Context
Quantum approximate optimization (QAOA) has been proposed for finding Hadamard matrices. The GS array structure reduces the qubit count from O(n²) to O(n).

## Qubit Requirements
- Full search: 668² = 446,224 qubits (far beyond current hardware)
- GS-array: 4 × 167 = 668 qubits (future near-term feasibility)
- Sub-problem: 167 qubits per sequence (marginal current feasibility)
- Current NISQ limit: ~50 effective qubits

## Implementation Backlog
- [ ] Formulate QUBO for single-sequence sub-problem (167 variables)
- [ ] Test on small instances (p=7: 28 qubits)
- [ ] Implement QAOA circuit in Qiskit
- [ ] Benchmark against classical SA
- [ ] Explore hybrid quantum-classical decomposition

## References
- Suksmono (2024), "A quantum approximate optimization method for finding Hadamard matrices"
- Suksmono & Minato (2019), "Finding Hadamard Matrices by a Quantum Annealing Machine"
