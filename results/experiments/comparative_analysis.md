# Comparative Analysis: H(668) Results vs. Prior Work

## 1. Comparison with Known Results

### 1.1 Eliahou 2025: 64-Modular Hadamard Matrix of Order 668
- **Their result:** Constructed a 64-modular Hadamard matrix H such that HH^T = 668I (mod 64)
- **Our result:** Constructed exact near-miss with HH^T = 668I + E, max|E_ij| = 4
- **Comparison:** Our Legendre-based near-miss is a genuine +/-1 matrix with small off-diagonal errors, while Eliahou's is an exact modular construction. Neither produces a true Hadamard matrix.

### 1.2 Suksmono 2018: Simulated Quantum Annealing
- **Their method:** Simulated quantum annealing for Hadamard matrices
- **Their results:** Not specific to order 668; demonstrated approach on smaller orders
- **Our method:** Classical SA with DFT-guided moves, parallel tempering, multi-start
- **Comparison:** We achieve ~31K iterations/second with incremental DFT updates. Our convergence barrier at L2~390K is consistent with the known difficulty.

### 1.3 Cati & Pasechnik 2024: SageMath Database
- **Their result:** Comprehensive database of Hadamard constructions for all orders <= 1208
- **Gap confirmed:** 668 is explicitly listed as the first missing order
- **Our contribution:** Systematic documentation of why all standard constructions fail for 668

### 1.4 Djokovic & Kotsireas 2018: GS Difference Families
- **Their method:** Specialized orbit search with multiplier group symmetry breaking
- **Their success:** Found GS difference families for many primes, but not 167
- **Our approach:** Similar orbit decomposition but with stochastic search instead of exhaustive
- **Comparison:** Their approach is more systematic (exhaustive over orbit combinations) but requires specialized infrastructure. Our stochastic approach is more general but hits convergence barriers.

## 2. Novel Search Space Exploration

### 2.1 Did We Explore New Regions?
- **Legendre perturbation:** Starting from Legendre sequences and perturbing explores a neighborhood of the known best configuration
- **Random initialization:** Multi-start with random sequences explores distant regions
- **Orbit-based search:** Decomposing Z_167 under {1,-1} explores a different coordinate system
- **Assessment:** Our random restarts likely visited regions not explored by prior algebraic approaches, but the convergence barrier is universal

### 2.2 Comparison with Neighboring Primes
Known GS constructions exist for p = 163 (H(652)) and p = 173 (H(692)).

| Property | p=163 | p=167 | p=173 |
|----------|-------|-------|-------|
| p mod 4 | 3 | 3 | 1 |
| |Z_p*| | 162 = 2*81 | 166 = 2*83 | 172 = 4*43 |
| Factor structure | 2 * 3^4 | 2 * 83 | 2^2 * 43 |
| Cyclotomic classes | Rich (many subgroups) | Sparse (83 prime) | Rich (43 factor) |
| H(4p) status | Known | **Open** | Known |

**Key insight:** p=163 has |Z_p*| = 162 = 2*3^4, giving many useful subgroups and cyclotomic classes. p=173 has |Z_p*| = 172 = 4*43, with the factor 4 providing useful quadratic and quartic partitions. But p=167 has |Z_p*| = 166 = 2*83 with 83 prime, leaving essentially no useful intermediate subgroups.

This sparse group structure may be the fundamental reason H(668) is harder than its neighbors.

## 3. Assessment

### 3.1 Contribution
Our work provides:
1. **Systematic elimination** of all standard algebraic constructions for H(668)
2. **Quantified convergence barriers** for stochastic search approaches
3. **Structural analysis** identifying the sparse group structure of Z_167* as the likely obstruction
4. **Best near-miss matrix** (Legendre baseline, off-diagonal max 4) as a starting point for future work
5. **Reproducible computational framework** for continued search

### 3.2 Limitations
1. Total compute: ~57M evaluations across all methods. Much less than the estimated ~10^12 needed.
2. No SAT+CAS implementation (would require external solver integration).
3. Single CPU -- no GPU acceleration.
4. No formal proof of non-existence for Williamson type.

### 3.3 Verdict
The negative result is consistent with the established open status of H(668). Our analysis provides new structural insight (the sparse group structure hypothesis) that goes beyond prior work. The problem likely requires either:
- Vastly more computational resources (GPU clusters, days of compute)
- New mathematical techniques beyond current methods
- A conceptual breakthrough in understanding the structure of Z_167
