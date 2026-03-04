# Genetic Algorithm with Circulant Cores for H(668)

## Context
Kotsireas and Koukouvinos (2005) pioneered genetic algorithms for Hadamard matrix construction with two-circulant cores. Ruiz (2022) achieved GPU acceleration. For order 668 = 4×167, the GS array with circulant blocks is parameterized by four binary vectors of length 167.

## Key Advantage
The GA compresses the search from a 668×668 matrix (446,224 variables) to four 167-bit strings (668 variables), a 667× reduction. FFT-based autocorrelation evaluation makes fitness computation O(v log v) per block.

## Implementation Backlog
- [ ] Implement FFT-based autocorrelation fitness for Z_167
- [ ] Design chromosome encoding for 4 circulant blocks
- [ ] Implement GPU kernel for parallel fitness evaluation
- [ ] Test with known SDS (v=23, v=47)
- [ ] Cyclotomic-aware initialization
- [ ] Add orbit-aware crossover operators
- [ ] Run large-scale search for v=167
- [ ] Compare with propus-constrained variant
