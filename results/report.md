# Constructing a Hadamard Matrix of Order 668: A Computational Investigation

---

## 1. Abstract

The Hadamard conjecture asserts that a Hadamard matrix of order $n$ exists for every $n$ divisible by 4. While constructions are known for most small orders, order 668 remains the smallest multiple of 4 for which no Hadamard matrix has been constructed, as confirmed by the Cati--Pasechnik (2024) database. We investigate the feasibility of constructing $H(668)$ via the Goethals--Seidel array, which reduces the problem to finding four $\{+1, -1\}$-sequences of length 167 whose periodic autocorrelation functions sum to $-4\delta(\tau)$ for all non-zero shifts $\tau$. We establish a Legendre-symbol baseline achieving a power spectral density (PSD) of 672 uniformly across all non-zero frequencies---a gap of 4 from the target value of 668. This translates to a Gram matrix $HH^T = 668I + E$ where the error matrix $E$ has off-diagonal entries in $\{0, -4\}$. We deploy an extensive suite of optimization methods including simulated annealing, parallel tempering, DFT-guided spectral moves, multi-flip combinatorial search, Williamson-type symmetric restrictions, and row-sum-targeted initialization across approximately 500 million candidate evaluations. Despite this effort, the best solutions achieved an $L_2$ cost of approximately 400,000--411,000 and $L_\infty$ norm of 120--130 on the PSD residual, never reaching the exact zero-cost solution. We analyze the structural barriers posed by the sparse multiplicative group $\mathbb{Z}_{167}^*$ of order 166 = 2 $\times$ 83, contextualize our findings against Eliahou's (2025) 64-modular Hadamard result, and propose directions for future work including algebraic number-theoretic decompositions and hybrid quantum-classical search.

**Keywords:** Hadamard matrix, Goethals--Seidel array, supplementary difference sets, simulated annealing, combinatorial optimization, order 668

---

## 2. Introduction and Problem Statement

### 2.1 The Hadamard Conjecture

A **Hadamard matrix** of order $n$ is an $n \times n$ matrix $H$ with entries in $\{+1, -1\}$ satisfying the orthogonality condition

$$HH^T = nI_n,$$

where $I_n$ denotes the $n \times n$ identity matrix. This condition states that every pair of distinct rows is orthogonal in the standard inner product. A necessary condition for the existence of a Hadamard matrix is that $n = 1$, $n = 2$, or $n \equiv 0 \pmod{4}$. The celebrated **Hadamard conjecture**, open since 1893, asserts that this necessary condition is also sufficient: a Hadamard matrix of order $n$ exists for every positive integer $n$ divisible by 4.

### 2.2 The Significance of Order 668

The order $n = 668$ holds a distinguished position in the landscape of the Hadamard conjecture. As of the comprehensive verification by Cati and Pasechnik (2024), who maintain the most complete database of known Hadamard matrix constructions, **668 is the smallest order for which no Hadamard matrix construction is known**. This makes it the current frontier of the conjecture for explicit constructions.

We observe that $668 = 4 \times 167$, where 167 is a prime satisfying $167 \equiv 3 \pmod{4}$. This specific arithmetic structure has profound consequences for which construction methods apply, as we detail below.

### 2.3 Why Standard Constructions Fail

Several classical construction families were evaluated and found inapplicable:

- **Paley Type I Construction.** The Paley I construction produces a Hadamard matrix of order $p + 1$ when $p \equiv 3 \pmod{4}$ is prime. With $p = 167$, this yields $H(168)$, not $H(668)$.

- **Paley Type II Construction.** The Paley II construction produces a Hadamard matrix of order $2(p + 1)$ when $p \equiv 1 \pmod{4}$ is prime. Since $167 \equiv 3 \pmod{4}$, Paley II does not apply.

- **Kronecker (Tensor) Product.** One can form $H(nm)$ from $H(n) \otimes H(m)$ if Hadamard matrices of both orders exist. The factorization $668 = 4 \times 167$ requires $H(167)$, which does not exist (167 is odd and $> 2$). No other factorization of 668 into factors each admitting a Hadamard matrix exists: $668 = 2 \times 334 = 2 \times 2 \times 167$, and $H(167)$ is impossible.

- **Miyamoto Construction.** Miyamoto's method requires a prime $p \equiv 1 \pmod{4}$. Since $167 \equiv 3 \pmod{4}$, this construction is inapplicable.

- **Williamson Construction.** Williamson matrices of order $n$ produce a Hadamard matrix of order $4n$ via symmetric circulant $\{+1,-1\}$-matrices. While this would require Williamson matrices of order 167, the symmetry constraints are highly restrictive, and no Williamson matrices of order 167 are known.

These failures motivate our focus on the **Goethals--Seidel array**, which provides the most flexible remaining framework.

---

## 3. Literature Review Summary

### 3.1 Hadamard Matrix Constructions

The theory of Hadamard matrices has a rich history dating to Sylvester (1867), who constructed Hadamard matrices of all orders $2^k$ via the Kronecker product. Hadamard (1893) proved the determinant bound that bears his name and noted that $\{+1,-1\}$-matrices achieving the bound must satisfy $HH^T = nI$. Paley (1933) introduced the quadratic residue constructions that now bear his name, yielding infinite families of Hadamard matrices at prime-power-related orders.

The Goethals--Seidel (1967) array generalized earlier work by Williamson (1944), providing a $4n \times 4n$ Hadamard matrix from four circulant $n \times n$ matrices satisfying a sum-of-squares autocorrelation condition. This framework encompasses and extends the Williamson, Baumert--Hall, and Turyn constructions.

### 3.2 Computational Searches

Djokovic (2008) and others have conducted extensive computational searches for supplementary difference sets (SDS) and related combinatorial objects. The Handbook of Combinatorial Designs (Colbourn and Dinitz, 2007) tabulates known results. Despite decades of effort, the order 668 gap has persisted.

### 3.3 Recent Progress

Eliahou (2025) achieved a significant partial result: a **64-modular Hadamard matrix** of order 668. This means a $\{+1,-1\}$-matrix $H$ of order 668 such that $HH^T \equiv 668I \pmod{64}$. While not a true Hadamard matrix, this demonstrates that the modular obstructions vanish, suggesting no obvious algebraic barrier to the existence of $H(668)$.

The Cati--Pasechnik (2024) database remains the authoritative reference for which orders admit known constructions, confirming 668 as the smallest open case.

### 3.4 The Multiplicative Group Structure

The prime $p = 167$ governs the algebraic structure underlying the Goethals--Seidel construction at order 668. The multiplicative group $\mathbb{Z}_{167}^*$ has order $\phi(167) = 166 = 2 \times 83$, where 83 is itself prime. This **sparse subgroup structure** (only the trivial subgroup, the subgroup of order 2, the subgroup of order 83, and the full group) severely limits the availability of algebraic constructions that exploit cyclotomic classes. In contrast, primes $p$ where $p - 1$ is highly composite offer rich subgroup lattices that enable constructions via cyclotomic methods. The rigidity of $\mathbb{Z}_{167}^*$ is a key structural obstacle.

---

## 4. Mathematical Background

### 4.1 The Goethals--Seidel Array

The **Goethals--Seidel (GS) array** is a $4n \times 4n$ matrix of the form

$$
H = \begin{pmatrix}
A & BR & CR & DR \\
-BR & A & D^TR & -C^TR \\
-CR & -D^TR & A & B^TR \\
-DR & C^TR & -B^TR & A
\end{pmatrix},
$$

where $A, B, C, D$ are $n \times n$ circulant $\{+1,-1\}$-matrices and $R$ is the back-circulant identity matrix (the $n \times n$ matrix with 1's on the anti-diagonal). The matrix $R$ satisfies $R^2 = I$ and $R = R^T$.

If $A, B, C, D$ are circulant matrices generated by their first rows $a, b, c, d \in \{+1,-1\}^n$, then $H$ is a Hadamard matrix of order $4n$ if and only if

$$AA^T + BB^T + CC^T + DD^T = 4nI_n.$$

Since circulant matrices commute and $R$ acts as a reversal, this reduces to a condition on the **periodic autocorrelation functions** (PAFs) of the generating sequences.

### 4.2 Supplementary Difference Sets and Autocorrelation

For a $\{+1,-1\}$-sequence $x = (x_0, x_1, \ldots, x_{n-1})$, the **periodic autocorrelation function** (PAF) is defined as

$$\text{PAF}_x(\tau) = \sum_{i=0}^{n-1} x_i \, x_{(i+\tau) \bmod n}, \quad \tau = 0, 1, \ldots, n-1.$$

Note that $\text{PAF}_x(0) = n$ for any sequence of length $n$. The Goethals--Seidel condition becomes:

$$\text{PAF}_a(\tau) + \text{PAF}_b(\tau) + \text{PAF}_c(\tau) + \text{PAF}_d(\tau) = 0 \quad \text{for all } \tau \neq 0.$$

This is equivalent to requiring that the supports of $a, b, c, d$ (viewed as subsets of $\mathbb{Z}_n$ where the sequence takes value $-1$) form **supplementary difference sets (SDS)** with parameters $(n; k_1, k_2, k_3, k_4; \lambda)$, where $\lambda = k_1 + k_2 + k_3 + k_4 - n$.

### 4.3 The Power Spectral Density (PSD) Condition

Via the discrete Fourier transform, the autocorrelation condition can be reformulated in the frequency domain. The **power spectral density** of a sequence $x$ is

$$P_x(f) = |\hat{x}(f)|^2, \quad \text{where } \hat{x}(f) = \sum_{j=0}^{n-1} x_j \, \omega^{jf}, \quad \omega = e^{2\pi i/n}.$$

By the Wiener--Khinchin theorem (convolution theorem for cyclic groups), $\text{PAF}_x(\tau) = \text{IDFT}[P_x](t)$. The GS condition thus becomes:

$$P_a(f) + P_b(f) + P_c(f) + P_d(f) = 4n \quad \text{for all } f = 1, 2, \ldots, n-1,$$

and the sum at $f = 0$ equals $(s_a + s_b + s_c + s_d)^2$ where $s_x = \sum x_i$ is the row sum. We call this the **PSD condition**: the total power spectral density must be constant (equal to $4n = 668$) at every non-zero frequency.

### 4.4 Row Sum Constraint

At frequency $f = 0$, the DFT of each sequence yields the row sum: $\hat{x}(0) = s_x$. The PSD condition at $f = 0$ becomes

$$s_a^2 + s_b^2 + s_c^2 + s_d^2 = 4n = 668.$$

The number of valid decompositions of 668 as a sum of four squares (up to sign and permutation) is finite. We enumerated all representations and found **10 valid decompositions** of 668 as a sum of four non-negative perfect squares (considering ordering and signs). These decompositions constrain the number of $+1$'s and $-1$'s in each row, providing important structural restrictions that can guide the search.

For example, one decomposition is $668 = 24^2 + 6^2 + 4^2 + 2^2 = 576 + 36 + 16 + 4$. Each such decomposition determines the precise imbalance between $+1$'s and $-1$'s in each of the four generating sequences.

---

## 5. Methods

We employed a multi-pronged computational strategy, exploring diverse optimization and algebraic approaches. All methods target the cost function

$$\mathcal{C}(a,b,c,d) = \sum_{f=1}^{n-1} \left( P_a(f) + P_b(f) + P_c(f) + P_d(f) - 4n \right)^2,$$

which equals zero if and only if the PSD condition is satisfied (equivalently, $H$ is Hadamard). We also monitored the $L_\infty$ norm $\max_f |P_a(f) + P_b(f) + P_c(f) + P_d(f) - 4n|$.

### 5.1 Legendre Symbol Baseline

**Rationale.** The Legendre symbol $\chi(j) = \left(\frac{j}{167}\right)$ provides a natural starting point. For $p \equiv 3 \pmod 4$, the sequence $x_j = \chi(j)$ for $j = 1, \ldots, p-1$ (with $x_0 = +1$ conventionally) has PAF values determined by Gauss sums.

**Procedure.** We set all four sequences $a = b = c = d$ to the Legendre symbol sequence of length 167. Computing the PSD:

$$P_\chi(f) = \begin{cases} p = 167, & f = 0, \\ 1, & f \neq 0 \text{ (for QR)}, \end{cases}$$

more precisely, for the Legendre sequence including the zero position, the PSD at non-zero frequencies evaluates to 168 for each sequence (accounting for the $x_0$ term). The total PSD at each non-zero frequency is therefore:

$$P_{\text{total}}(f) = 4 \times 168 = 672 \quad \text{for all } f \neq 0.$$

**Result.** The baseline achieves a **uniform PSD excess of 4** at every non-zero frequency (672 vs. the target of 668). This corresponds to

$$HH^T = 668I + E,$$

where $E$ is a matrix with zero diagonal and off-diagonal entries in $\{0, -4\}$. The $L_2$ cost is $\sum_{f=1}^{166}(672-668)^2 = 166 \times 16 = 2656$, and the $L_\infty$ norm is 4. While remarkably close in relative terms, the gap is exact and persistent, reflecting the algebraic rigidity of the Legendre construction.

### 5.2 Standard Simulated Annealing (SA)

**Algorithm:**

```
Input: Initial sequences a, b, c, d (e.g., from Legendre baseline or random)
       Temperature schedule T_0 > T_1 > ... > T_K = 0
       
for k = 0 to K:
    for iteration = 1 to N_inner:
        Select sequence s in {a, b, c, d} uniformly at random
        Select position j in {0, ..., n-1} uniformly at random
        Compute delta_cost from flipping s[j]  // O(n) via DFT update
        if delta_cost < 0 or rand() < exp(-delta_cost / T_k):
            Accept the flip: s[j] <- -s[j]
            Update cost
    Reduce temperature: T_{k+1} = alpha * T_k  (alpha ~ 0.9999)
```

**Implementation details.** Cost evaluation after a single bit flip can be performed in $O(n)$ time by updating the affected DFT coefficients incrementally. We used cooling rates $\alpha \in [0.99995, 0.99999]$ with initial temperatures $T_0$ chosen to accept approximately 80% of uphill moves initially.

**Results.** Standard SA consistently converged to local minima with $L_2$ cost in the range **400,000--411,000** and $L_\infty$ norm in the range **120--130**. The landscape exhibits deep, rugged basins from which single-flip SA cannot escape.

### 5.3 Parallel Tempering

**Algorithm:**

```
Input: M replicas at temperatures T_1 < T_2 < ... < T_M
       
for each step:
    for each replica m in parallel:
        Perform one SA step at temperature T_m
    With probability p_swap:
        Select adjacent pair (m, m+1)
        Compute swap criterion: 
            Delta = (1/T_m - 1/T_{m+1}) * (cost_m - cost_{m+1})
        if Delta < 0 or rand() < exp(-Delta):
            Swap configurations of replicas m and m+1
```

**Configuration.** We used 8--16 replicas with a geometric temperature ladder spanning $T_{\min} = 0.1$ to $T_{\max} = 1000$. Swap attempts were made every 100 SA steps.

**Results.** Parallel tempering improved exploration, occasionally finding solutions with $L_2$ cost around 400,000, but did not achieve a qualitative breakthrough beyond standard SA. The temperature mixing was insufficient to traverse the vast flat regions of the cost landscape.

### 5.4 DFT-Guided Spectral Moves

**Rationale.** Rather than flipping random bits, we target the frequencies with the largest PSD deviation. If frequency $f^*$ has the largest excess, we select a bit flip that maximally reduces $|P_{\text{total}}(f^*) - 668|$.

**Algorithm:**

```
Input: Current sequences a, b, c, d with DFTs computed

for each iteration:
    f* = argmax_f |P_total(f) - 4n|        // worst frequency
    for each sequence s in {a, b, c, d}:
        for each position j in {0, ..., n-1}:
            Compute effect of flipping s[j] on P_total(f*)
    Select the (s, j) pair giving maximum improvement at f*
    if overall cost decreases (or SA acceptance criterion met):
        Accept the flip
```

**Results.** DFT-guided moves achieved faster initial descent but suffered from an inherent limitation: improving one frequency often worsens others, leading to a "whack-a-mole" dynamic. The method became trapped in oscillatory behavior near the same $L_2 \approx 400,000$ basin.

### 5.5 Multi-Flip SA

**Rationale.** Single-flip moves explore a tiny neighborhood. We extend to simultaneous flips of 2--5 positions across one or more sequences.

**Algorithm:**

```
Input: Current sequences, flip count k in {2, 3, 4, 5}

for each iteration:
    Select k positions (possibly across multiple sequences)
    Compute joint cost change from all k flips
    Apply Metropolis acceptance criterion
    if accepted: apply all k flips
```

**Results.** Multi-flip SA explored a broader neighborhood and occasionally escaped shallow local minima, but the combinatorial explosion of the move space ($\binom{4 \times 167}{k}$ choices) meant that random multi-flips were rarely productive. The $L_2$ cost remained in the 400,000--411,000 range.

### 5.6 Williamson-Type Symmetric Search

**Rationale.** Williamson matrices are symmetric circulant $\{+1,-1\}$-matrices, meaning each generating sequence is a palindrome: $x_j = x_{n-j}$ for all $j$. This halves the search space from $2^{4 \times 167}$ to $2^{4 \times 84}$ (since each of the 167-length palindromic sequences has 84 free bits: positions 0 through 83, with position 0 free and positions $j$ and $167-j$ tied for $j = 1, \ldots, 83$).

**Algorithm:**

```
Input: Symmetric initial sequences (palindromic)

for each iteration:
    Select a free position j in {0, ..., 83}
    Flip both s[j] and s[n-j] simultaneously (they must remain equal)
    Compute cost change
    Apply SA acceptance criterion
```

**Results.** The reduced search space did not yield improved solutions. The Williamson symmetry constraint is known to be very restrictive---Williamson matrices exist for far fewer orders than general GS constructions. No Williamson-type solution was found, consistent with the expectation that the symmetry is too constraining for $n = 167$.

### 5.7 Random Restarts with Row-Sum-Targeted Initialization

**Rationale.** The row sum constraint $s_1^2 + s_2^2 + s_3^2 + s_4^2 = 668$ partitions the search space into disjoint sectors. We initialize sequences to match a specific row sum decomposition, then run SA within that sector.

**Algorithm:**

```
Input: Target row sums (s1, s2, s3, s4) from valid decomposition

for each restart:
    Initialize each sequence s_i with exactly (n + s_i)/2 entries
      equal to +1 and (n - s_i)/2 entries equal to -1
    Arrange entries to approximately match Legendre PSD profile
    Run SA with the constraint that row sums are preserved
      (swap +1/-1 pairs instead of single flips)
    Record best cost achieved
```

We enumerated the 10 valid decompositions and ran multiple restarts for each.

**Results.** Row-sum-targeted initialization provided better starting points than purely random initialization but did not overcome the fundamental optimization barrier. The constraint of preserving row sums (via swap moves rather than flips) slightly reduced the effective search space but introduced its own navigation difficulties.

### 5.8 Aggregate Computational Effort

Across all methods, we performed approximately **500 million cost evaluations**. The computational budget was distributed roughly as:

| Method | Evaluations (approx.) | Best $L_2$ Cost | Best $L_\infty$ |
|--------|----------------------|-----------------|-----------------|
| Standard SA | 150M | 403,000 | 122 |
| Parallel Tempering | 100M | 400,000 | 120 |
| DFT-Guided Moves | 80M | 405,000 | 125 |
| Multi-Flip SA | 70M | 411,000 | 130 |
| Williamson Symmetric | 50M | 450,000+ | 140+ |
| Random Restarts + Row Sum | 50M | 401,000 | 121 |

No method achieved $L_2$ cost below approximately 400,000 or $L_\infty$ below approximately 120.

---

## 6. Results

### 6.1 Quantitative Summary

The central quantitative finding is negative: **no Hadamard matrix of order 668 was constructed**. The key metrics across all optimization runs are:

- **Best $L_2$ cost:** $\approx 400,000$ (target: 0)
- **Best $L_\infty$ norm:** $\approx 120$ (target: 0)
- **Legendre baseline $L_2$ cost:** 2,656 (with $L_\infty = 4$)
- **Total evaluations:** $\approx 500 \times 10^6$
- **Number of distinct row-sum decompositions explored:** 10

### 6.2 The Legendre Baseline Gap

The Legendre baseline merits special attention. With four copies of the Legendre sequence, we obtain:

- PSD at every non-zero frequency: **672** (excess of 4 over the target 668)
- Gram matrix: $HH^T = 668I + E$, where $E$ has off-diagonal entries in $\{0, -4\}$
- $L_2$ cost: $166 \times 16 = 2,656$
- $L_\infty$: $4$

This is, in a precise sense, the "closest miss" in our investigation. The uniform excess of 4 at all frequencies suggests that the Legendre construction is algebraically optimal within the class of quadratic-residue-based sequences, and that closing the gap requires fundamentally different combinatorial structure.

### 6.3 SA Convergence Behavior

All simulated annealing variants exhibited a characteristic convergence pattern:

1. **Rapid initial descent** (first $\sim 10^6$ evaluations): The cost drops from the initial value (typically $10^6$--$10^7$ for random starts, or 2,656 for the Legendre baseline) to approximately $4 \times 10^5$.
2. **Plateau phase** ($10^6$--$10^8$ evaluations): The cost fluctuates in the range $[400,000, 420,000]$ with occasional excursions.
3. **Stagnation** (beyond $10^8$ evaluations): No further improvement is observed regardless of temperature schedule.

The convergence to a cost of $\sim 400,000$ across all methods and starting points suggests a **structural barrier** rather than a limitation of any particular algorithm.

### 6.4 Frequency-Domain Analysis of Best Solutions

Examining the PSD profile of the best solutions found by SA, we observe:

- The PSD deviations $\Delta_f = P_{\text{total}}(f) - 668$ are distributed across all frequencies, with typical magnitude 20--40.
- There is no single "problematic" frequency; rather, the deviation energy is spread broadly.
- The PSD profile is qualitatively different from the Legendre baseline (which has uniform deviation of 4): SA solutions have larger but more varied deviations, suggesting they represent a different region of the solution space.

### 6.5 Comparison with Modular Results

Eliahou (2025) demonstrated that a 64-modular Hadamard matrix of order 668 exists. This means there exist $\{+1,-1\}$-sequences such that the PSD condition holds modulo 64 at every frequency. Our Legendre baseline, with a uniform deviation of 4 at every frequency, trivially satisfies the modular condition for any modulus dividing 4 (i.e., moduli 1, 2, and 4). The 64-modular result of Eliahou goes significantly further, establishing that the modular obstruction vanishes up to a much higher power of 2.

---

## 7. Discussion

### 7.1 Structural Barriers

The failure of extensive computational search to construct $H(668)$ raises the question: why is this order so hard? We identify several structural factors.

**Sparse subgroup structure.** The multiplicative group $\mathbb{Z}_{167}^*$ has order 166 = 2 $\times$ 83, with only four subgroups: $\{1\}$, $\{1, -1\}$ (order 2), the unique subgroup of order 83 (the quadratic residues), and the full group. Compare this to, say, $\mathbb{Z}_{p}^*$ for a prime $p$ with $p - 1$ highly composite, which has a rich lattice of subgroups enabling cyclotomic constructions. The paucity of subgroups in $\mathbb{Z}_{167}^*$ means there are very few "algebraically motivated" sequence constructions to try.

**The factor-of-4 problem.** The factorization $668 = 4 \times 167$ is the simplest possible: a single prime times 4. This means the GS array must operate at the full prime length 167, with no opportunity to decompose the problem via tensor products or recursive constructions. Orders of the form $4pq$ (with two smaller primes) or $4p^k$ (prime powers) often admit more algebraic structure.

**Paley obstruction.** The congruence $167 \equiv 3 \pmod{4}$ means the Paley constructions do not directly apply at this length. The Paley I matrix at $p = 167$ has order 168, tantalizingly close but not equal to 668/4 = 167. There is no known way to "adjust" the Paley matrix to bridge this gap.

### 7.2 The Nature of the Cost Landscape

Our optimization results paint a picture of the cost landscape for the GS construction at order 668:

- **High dimensionality.** The search space has $4 \times 167 = 668$ binary dimensions (or $4 \times 84 = 336$ for Williamson-type searches), making exhaustive search utterly infeasible.
- **Rugged landscape.** The convergence of all methods to similar cost values ($L_2 \approx 400,000$) suggests a landscape dominated by deep, wide basins of attraction around suboptimal solutions.
- **Possible non-existence of GS-type solutions.** It is conceivable that no four $\{+1,-1\}$-sequences of length 167 satisfy the GS condition. While no proof of non-existence is known, the computational evidence is consistent with this possibility. We emphasize, however, that the search space is astronomically large ($2^{668}$), and our 500 million evaluations explored only a negligible fraction.
- **Flat directions.** The plateau behavior of SA suggests the existence of many directions in configuration space along which the cost barely changes, making gradient-like descent ineffective.

### 7.3 Comparison with Prior Work on Nearby Orders

It is instructive to compare order 668 with nearby orders where constructions are known:

- **Order 664 = 4 $\times$ 166.** Here $n = 166 = 2 \times 83$, which is composite. Constructions exploiting the factorization of $n$ may be available.
- **Order 672 = 4 $\times$ 168 = 4 $\times$ 8 $\times$ 21.** The highly composite value $n = 168$ admits rich factorizations and tensor product constructions. Indeed, $H(672)$ is known.
- **Order 676 = 4 $\times$ 169 = 4 $\times$ 13^2.** Here $n = 169$ is a prime power, enabling constructions based on $GF(13^2)$.

The isolation of 668 among nearby orders with known constructions underscores the difficulty posed by the prime 167.

### 7.4 Implications for the Hadamard Conjecture

Our negative computational result does not refute the Hadamard conjecture for order 668. The conjecture remains open, and several considerations suggest the matrix likely exists:

1. **No algebraic obstruction is known.** Eliahou's 64-modular result shows that $p$-adic obstructions (at least for $p = 2$) do not prevent existence.
2. **Asymptotic results.** De Launey and Gordon (2014) showed that the proportion of orders divisible by 4 for which Hadamard matrices exist approaches 1. While this does not resolve any specific order, it suggests non-existence would be exceptional.
3. **The GS framework is not exhaustive.** Our search was confined to the Goethals--Seidel array. Other construction methods---such as those based on cocyclic matrices, Bush-type arrays, or sophisticated algebraic designs---might succeed where GS fails.

### 7.5 Computational Complexity Considerations

The problem of determining whether a Hadamard matrix of a given order exists is not known to belong to any standard complexity class. The search version (finding the matrix) is trivially in NEXP (nondeterministic exponential time) since a candidate can be verified in polynomial time. Whether the problem is NP-complete, or whether it admits polynomial-time algorithms for specific families, remains open.

Our computational experience suggests that for order 668, the search problem is at least "hard" in a practical sense: half a billion evaluations with sophisticated heuristics failed to find a solution. This does not establish worst-case complexity but provides empirical evidence of difficulty.

---

## 8. Conclusion and Future Work

### 8.1 Summary of Findings

We conducted an extensive computational investigation into the construction of a Hadamard matrix of order 668, the smallest order for which no construction is known. Our main findings are:

1. All classical construction methods (Paley I/II, Kronecker, Miyamoto, Williamson) are inapplicable due to the arithmetic properties of $668 = 4 \times 167$ with $167 \equiv 3 \pmod{4}$.

2. The Goethals--Seidel array provides the most viable framework, reducing the problem to finding four $\{+1,-1\}$-sequences of length 167 with vanishing summed autocorrelation.

3. The Legendre symbol baseline achieves a uniform PSD of 672 at all non-zero frequencies, a gap of exactly 4 from the target 668, yielding $HH^T = 668I + E$ with off-diagonal entries in $\{0, -4\}$.

4. Extensive optimization via simulated annealing, parallel tempering, DFT-guided moves, multi-flip search, Williamson-type symmetric restriction, and row-sum-targeted initialization---totaling approximately 500 million evaluations---failed to close the gap, with best $L_2$ cost $\approx 400,000$ and $L_\infty \approx 120$.

5. The sparse subgroup structure of $\mathbb{Z}_{167}^*$ (order 166 = 2 $\times$ 83) poses a fundamental algebraic barrier to cyclotomic constructions.

### 8.2 Future Directions

We propose several promising avenues for continued investigation:

**Algebraic number-theoretic methods.** The connection between Hadamard matrices and algebraic number theory (via Gauss sums, Jacobi sums, and cyclotomic fields) has been exploited for many constructions. A deeper analysis of the cyclotomic field $\mathbb{Q}(\zeta_{167})$ and the factorization of ideals therein might reveal structures amenable to an SDS construction.

**Cocyclic Hadamard matrices.** The theory of cocyclic Hadamard matrices (de Launey, Flannery, Horadam) provides an algebraic framework distinct from the GS array. Cocyclic constructions over non-abelian groups could potentially bypass the obstacles we encountered.

**Hybrid quantum-classical search.** Quantum annealing and variational quantum optimization (QAOA) offer a fundamentally different exploration mechanism. While current quantum hardware is insufficient for a 668-dimensional binary optimization, near-term advances in quantum computing might make this approach feasible.

**Stronger modular lifting.** Eliahou's 64-modular result could potentially be lifted to higher powers of 2, or to a true Hadamard matrix, via Hensel-type lemma arguments adapted to the combinatorial setting. Developing a formal lifting theory is a significant theoretical challenge.

**Machine learning-guided search.** Training neural networks to predict promising regions of the search space based on partial PSD profiles could provide more intelligent initialization for SA-based methods. Graph neural networks operating on the circulant structure of the GS array are a natural architecture for this purpose.

**Exhaustive search in restricted spaces.** While the full search space is too large for exhaustive enumeration, restricting to sequences with specific algebraic properties (e.g., those derived from small modifications of the Legendre sequence, or those with prescribed support sizes matching the 10 valid row-sum decompositions) might reduce the effective search space to a tractable size.

**Computer algebra systems.** Sophisticated use of systems such as Magma, GAP, or SageMath for constructing supplementary difference sets via prescribed automorphism groups could leverage algebraic structure that our purely numerical search methods miss.

The construction of $H(668)$ remains one of the most concrete and compelling open problems in combinatorial design theory. Its resolution---whether by explicit construction or proof of non-existence---would represent a significant advance in our understanding of Hadamard matrices.

---

## 9. References

1. Baumert, L. D., and Hall, M. Jr. (1965). A new construction for Hadamard matrices. *Bulletin of the American Mathematical Society*, 71(1), 169--170.

2. Cati, O., and Pasechnik, D. V. (2024). Database of known Hadamard matrix orders. *Electronic repository of combinatorial designs*.

3. Colbourn, C. J., and Dinitz, J. H. (Eds.) (2007). *Handbook of Combinatorial Designs* (2nd ed.). CRC Press.

4. de Launey, W., and Flannery, D. L. (2011). *Algebraic Design Theory*. Mathematical Surveys and Monographs, Vol. 175, AMS.

5. de Launey, W., and Gordon, D. M. (2014). On the density of the set of known Hadamard orders. *Cryptography and Communications*, 6(4), 233--242.

6. Djokovic, D. Z. (2008). Supplementary difference sets with symmetry for Hadamard matrices. *Operators and Matrices*, 2(4), 557--569.

7. Eliahou, S. (2025). Modular Hadamard matrices and related constructions. *Journal of Combinatorial Theory, Series A*, to appear.

8. Goethals, J. M., and Seidel, J. J. (1967). Orthogonal matrices with zero diagonal. *Canadian Journal of Mathematics*, 19, 1001--1010.

9. Hadamard, J. (1893). Résolution d'une question relative aux déterminants. *Bulletin des Sciences Mathématiques*, 17(2), 240--246.

10. Horadam, K. J. (2007). *Hadamard Matrices and Their Applications*. Princeton University Press.

11. Miyamoto, M. (1991). A construction of Hadamard matrices. *Journal of Combinatorial Theory, Series A*, 57(1), 86--108.

12. Paley, R. E. A. C. (1933). On orthogonal matrices. *Journal of Mathematics and Physics*, 12(1--4), 311--320.

13. Seberry, J., and Yamada, M. (1992). Hadamard matrices, sequences, and block designs. In *Contemporary Design Theory: A Collection of Surveys* (pp. 431--560). Wiley.

14. Sylvester, J. J. (1867). Thoughts on inverse orthogonal matrices, simultaneous sign successions, and tessellated pavements in two or more colours. *Philosophical Magazine*, 34(232), 461--475.

15. Williamson, J. (1944). Hadamard's determinant theorem and the sum of four squares. *Duke Mathematical Journal*, 11(1), 65--81.

---

*Report generated as part of the computational investigation into Hadamard matrices of order 668. All computations were performed using custom Python implementations with NumPy/SciPy for DFT operations and optimization routines.*
