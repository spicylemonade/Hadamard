# Benchmark Report: GS Search Framework Performance

## Overview
This report benchmarks the GS search framework across multiple orders to estimate computational requirements for H(668).

## Test Configuration
- Seed: 42
- SA parameters: T_init=10.0, T_min=0.001, alpha=0.9999
- Move type: swap one element in/out of a random block
- Cost function: L2 norm of PSD deviation from target

## Results by Order

| Order n | p | Orbits | Iterations | Best L2 Cost | Best Linf | Time (s) | Found? |
|---------|---|--------|------------|-------------|-----------|----------|--------|
| 44 | 11 | 5 | 200,000 | 160.0 | 4.0 | 16.7 | No |
| 172 | 43 | 21 | 200,000 | 8,928 | 36.7 | 20.7 | No |
| 668 | 167 | 83 | 100,000/decomp | ~577,152 | ~147 | ~40 | No |

## Analysis

### Search Space Scaling
- p=11: search space 2^44 ~ 10^13 (but effective space much smaller with orbits)
- p=43: search space 2^172 ~ 10^52
- p=167: search space 2^668 ~ 10^201

### Convergence Behavior
The SA consistently converges to local minima that correspond to Legendre-like configurations. The flat PSD excess is characteristic: the algorithm reaches states where the PSD deviation is approximately constant across frequencies.

### Extrapolated Time for Exhaustive Search at v=167
With 83 orbits under {1,-1} and 4 blocks, the orbit-reduced search space is still approximately C(83, k)^4 for support sizes around k~40. This gives:
- C(83, 40) ~ 10^23
- Total: ~10^92 configurations
- At 10^5 evaluations/second: ~10^87 seconds

**Conclusion: Exhaustive orbit search is infeasible.** Stochastic methods remain the only viable approach, but convergence to local minima is a fundamental barrier.

### Computational Bottleneck
The bottleneck is FFT evaluation at each iteration: O(p log p) per evaluation. With p=167, each FFT takes ~10 microseconds. The swap-and-evaluate cycle achieves ~10K-50K iterations/second depending on implementation.

### Comparison with Prior Work
- Djokovic & Kotsireas (2018): reported finding GS difference families for primes up to ~200 using specialized search with multiplier group symmetry and multi-level orbit decomposition
- Suksmono (2018): simulated quantum annealing for Hadamard, no specific results at order 668
- Our results: consistent with the known difficulty of v=167

## Recommendations
1. Standard SA is insufficient -- need parallel tempering or population-based methods
2. DFT-guided moves (targeting worst-frequency flips) may help escape plateaus
3. Hybrid approaches (algebraic initialization + stochastic refinement) are most promising
4. Extended runs (10^8+ iterations) with periodic restarts needed
