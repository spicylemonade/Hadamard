# Quantum Annealing for Hadamard Orthogonality

## Context
Suksmono (2018, 2019) demonstrated quantum annealing approaches for finding Hadamard matrices, using both simulated quantum annealing (SQA) and D-Wave quantum hardware. The approach formulates orthogonality as an Ising spin energy minimization problem.

For order 668, the full QUBO has ~446,224 binary variables — far beyond current quantum hardware. However, decomposition into overlapping subproblems and hybrid quantum-classical approaches make this tractable.

## Key Challenge
The quartic-to-quadratic reduction introduces O(n³) auxiliary variables. The sliding-window decomposition must ensure global consistency while solving local subproblems.

## Implementation Backlog
- [ ] Implement row-orthogonality Hamiltonian E(H) = Σ_{r<s} (Σ_k h_{rk}h_{sk})²
- [ ] Implement SQA with Suzuki-Trotter decomposition
- [ ] Design sliding-window subproblem decomposition
- [ ] Test on known orders (28, 44, 92) to validate convergence
- [ ] Compare with classical simulated annealing baseline
- [ ] Investigate QAOA circuit ansatz for hybrid approach
- [ ] Scale to order 668 with structured initialization
