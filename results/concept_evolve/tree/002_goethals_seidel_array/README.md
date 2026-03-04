# Goethals-Seidel Array Construction

## Context
The GS array converts four ±1 matrices A,B,C,D of order n satisfying AA^T+BB^T+CC^T+DD^T=4nI into a Hadamard matrix of order 4n. For H(668), need n=167.

## Key Equations
- GS(A,B,C,D) is a 4n×4n matrix using anti-diagonal permutation R
- For circulant A,B,C,D: condition reduces to PAF flatness
- In frequency domain: sum_i |S_i(k)|^2 = 668 for all k

## Computational Results
- Random initialization + greedy descent: plateaus at energy ~500K
- Multi-start (10 trials): best energy = 486,304
- Cyclotomic initialization: energy ~598K (worse than random)
- This confirms the extreme difficulty of the open problem

## Implementation Backlog
- [x] Implement GS array construction
- [x] Implement FFT-based PSD computation
- [x] Run multi-start greedy optimization
- [x] Test cyclotomic initialization
- [ ] Implement simulated annealing with temperature schedule
- [ ] Try genetic algorithm / evolutionary approach
- [ ] Encode as SAT/SMT problem
- [ ] Test with constraint propagation

## References
- Goethals & Seidel (1970), "A skew Hadamard matrix of order 36"
- Djoković (2008), "Hadamard matrices of order 764 exist"
