# Spectral Flatness Condition

## Context
For the GS array with circulant matrices, the Hadamard condition becomes: the total power spectral density of 4 sequences must be flat at 668 across all frequencies.

## Mathematical Details
- Each ±1 sequence of length 167 has energy 167² = 27889 (Parseval)
- Total energy = 4 × 27889 = 111556
- Need: P(k) = 668 for all k = 0,...,166
- Verification: 668 × 167 = 111556 ✓

## Energy Landscape
- The objective E = sum (P(k) - 668)² has many local minima
- Greedy descent from random starts: plateaus at ~486K-600K
- The landscape resembles a spin glass
- Temperature-based methods (SA) help but don't reach E=0

## Implementation Backlog
- [x] Implement FFT-based PSD computation
- [x] Implement incremental PSD update for single flips
- [x] Run greedy optimization (10 restarts)
- [ ] Implement simulated annealing with Metropolis criterion
- [ ] Try gradient descent on continuous relaxation
- [ ] Analyze landscape topology (basin sizes, barrier heights)

## References
- Rudelson (2022), "Approximately Hadamard matrices and Riesz bases"
