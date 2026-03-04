# Repository Analysis for H(668) Construction

## Problem Statement
Find a Hadamard matrix of order 668 = 4 × 167. This is the smallest order for which no Hadamard matrix construction is known.

## Prior Attempts (Branch origin/research-lab-1772438840)
The prior branch contained two Python scripts:
1. `results/construct_hadamard_668.py` (~500 lines): Attempted various algebraic constructions (Paley, Goethals-Seidel with Legendre sequences, SDS-based approaches). All failed because:
   - Paley I gives H(168), not H(668)
   - Paley II gives H(336), not H(668)
   - Kronecker products cannot reach 668 (no factorization into valid Hadamard orders)
   - SDS candidates based on QR/QNR have intersection counts off by 1

2. `results/gs_search.py` (~400 lines): Attempted stochastic search for GS difference families using:
   - Orbit-based exhaustive search (QR/QNR orbits, 4096 combinations): no valid family found
   - Williamson stochastic search (100K iterations): converged to L2 ≈ 2.2M
   - General SA (200K iterations): insufficient iterations, non-convergence

### Why Prior Attempts Failed
- The PSD of 4 Legendre sequences is 672 at non-zero frequencies (target: 668), an excess of 4
- All simple algebraic constructions from QR/QNR sets have intersection counts off by 1
- SA with only 200K iterations is far too few for a search space of 2^668

## Reusable Components
- Legendre symbol computation
- Goethals-Seidel array builder
- FFT-based PSD checker
- Basic verification framework

## Key Mathematical Structure
- 167 is prime, 167 ≡ 3 (mod 4)
- Z_167* has order 166 = 2 × 83 (both prime)
- Gauss sum G = i√167 (pure imaginary)
- PSD of Legendre sequences is FLAT at 168 (= 1 + 167) for each sequence at k≠0
- 4 × 168 = 672 ≠ 668, the fundamental gap
