# Negative Results: H(668) Construction Attempts

## Summary
No Hadamard matrix of order 668 was found. This document systematically records all negative results and the convergence barriers encountered.

## 1. Methods Attempted and Best Results

### 1.1 Legendre Baseline (Best Linf = 4)
- **Sequences:** 4 copies of the Legendre symbol sequence for Z_167
- **PSD deviation:** Exactly +4 at all 166 non-zero frequencies
- **Structure:** This is the nearest known near-miss. The matrix HH^T = 668I + E where E has off-diagonal entries in {0, -4}
- **Why it fails:** Gauss sum for p=3 mod 4 gives |chi_hat(k)|^2 = p+1 = 168, so 4*168 = 672 != 668

### 1.2 General SA with Random Flips (Best L2 ~ 390K)
- **Method:** Simulated annealing with single-entry flips, incremental DFT update
- **Iterations:** 50M+ across 10 random restarts
- **Best L2 cost:** ~390,000 (sum of squared deviations)
- **Best Linf:** ~120 (max |PSD(k) - 668| at any frequency)
- **Convergence:** Rapid descent to L2 ~ 400K-600K within 200K iterations, then plateau
- **Rate:** ~31,000 iterations/second

### 1.3 Williamson-Restricted SA (Best L2 ~ 450K)
- **Method:** SA restricted to symmetric circulant sequences (84 independent positions each)
- **Iterations:** 5M
- **Best Linf:** ~130
- **Note:** Reduced search space (2^336 vs 2^668) but also reduced solution space

### 1.4 Parallel Tempering (Best L2 ~ 600K)
- **Method:** 4-8 temperature replicas with periodic swap attempts
- **Result:** No improvement over basic SA. Replica exchanges rare due to large cost differences.

### 1.5 DFT-Guided Moves (40x slower)
- **Method:** Select flip targeting the frequency with largest PSD deviation
- **Overhead:** Computing best guided flip takes ~2.2ms vs 0.05ms for random flip
- **Net effect:** Despite better per-move improvement, the 40x slowdown makes guided moves non-competitive in wall-clock time

### 1.6 Orbit-Based Search (Best L2 ~ 577K)
- **Method:** Construct supports from orbits under {1,-1} in Z_167
- **Result:** Orbit decomposition gives 83 orbits. Even with orbits, search space is vast.

### 1.7 Spence SDS (Not Computed)
- **Analysis:** Valid parameters 4-{334; 167,167,167,168; 334} in Z_334
- **Result:** Search space is 2x LARGER than GS approach. Not pursued computationally.

## 2. Landscape Analysis

### 2.1 Distribution of PSD Deviations
For the Legendre baseline, the PSD deviation is perfectly flat at +4 for all non-zero frequencies. This is the theoretical minimum for Legendre-based constructions.

For SA-optimized sequences, the PSD deviation is highly non-uniform:
- Some frequencies near target (deviation ~ 0-10)
- Others far from target (deviation ~ 100-200)
- The L2 cost is dominated by a few "bad" frequencies

### 2.2 Convergence Barrier
All methods encounter a convergence barrier around L2 ~ 390K-600K depending on initialization.
This corresponds to sequences where:
- ~50% of frequencies have PSD within 20 of target
- ~20% of frequencies have PSD deviation > 50
- A few frequencies have PSD deviation > 100

The barrier appears to be structural: improving one frequency typically worsens others due to the global constraint that total PSD is conserved.

### 2.3 Local Minima
SA consistently converges to distinct local minima. Multi-start experiments show:
- 10 random restarts produce 10 different local minima
- L2 costs range from 390K to 600K
- No pattern in which sequences lead to better minima

## 3. Structural Barriers

### 3.1 The Gap of 4
The fundamental barrier: for p = 3 (mod 4), the flat PSD of 4 Legendre sequences exceeds the target by exactly 4 at every frequency. Closing this gap requires finding sequences whose spectral properties differ from the Legendre sequence in a globally coordinated way.

### 3.2 No Useful Cyclotomic Classes
Z_167* has order 166 = 2 * 83. Since 83 is prime, the only subgroups are {1}, {1,-1}, the quadratic residues, and the full group. There are no intermediate cyclotomic classes that could provide a natural partitioning strategy.

### 3.3 Conservation Law
For +/-1 sequences of length p, the total spectral energy is conserved:
sum_k |a_hat(k)|^2 = p * sum_j a[j]^2 = p^2

This means reducing PSD at one frequency necessarily increases it at others. The PSD condition PSD(k) = 668 for all k requires a very precise balance.

## 4. Comparison with Neighboring Primes

| p | 4p | Status | GS construction known? |
|---|-----|--------|----------------------|
| 157 | 628 | KNOWN | Yes (in Cati-Pasechnik database) |
| 163 | 652 | KNOWN | Yes |
| 167 | 668 | **OPEN** | **No** |
| 173 | 692 | KNOWN | Yes |
| 179 | 716 | **OPEN** | **No** |

The fact that both neighboring primes 163 and 173 have known constructions but 167 and 179 do not is striking. Both 167 and 179 have sparse Z_p* structure (phi(167) = 2*83, phi(179) = 2*89, both with large prime factors), while 163 and 173 have richer subgroup lattices. This suggests the difficulty is related to the sparse multiplicative group structure.

## 5. Recommendations for Future Work

1. **SAT+CAS approach:** Encode the SDS problem directly as SAT with CAS-derived pruning. Estimated encoding: ~668 Boolean variables, ~O(p^2) clauses. May be tractable with modern solvers (CryptoMiniSat, Cadical).

2. **GPU-accelerated SA:** Our CPU-based SA achieves ~31K iter/s. GPU implementation could achieve ~10M iter/s, enabling 10^12+ evaluations. This may be sufficient to escape local minima.

3. **Machine learning guidance:** Train a neural network on known GS constructions for smaller primes to predict promising sequence patterns for p=167.

4. **Algebraic number theory:** Deeper analysis of the cyclotomic field Q(zeta_167) and its ideals may reveal structural constraints on SDS existence at v=167.

5. **Exhaustive search over restricted classes:** If Williamson-type H(668) does not exist, proving this would narrow the search to non-symmetric constructions.

## 6. Estimated Compute Requirements
- **To match Djokovic-Kotsireas level:** ~10^12 evaluations, ~1000 GPU-hours
- **For SAT approach:** Unknown, depends on propagation effectiveness
- **For proof of non-existence (Williamson):** ~2^336 ~ 10^101, currently infeasible
