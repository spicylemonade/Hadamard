# Cocyclic Cohomological Lift for H(668)

## Context
Horadam and de Launey (1993) introduced cocyclic development of designs, allowing Hadamard matrices to be classified by their algebraic structure via group cohomology. A cocyclic Hadamard matrix over G has entries M[g,h] = ψ(g,h) where ψ is a 2-cocycle in H²(G, Z₂).

For order 668, candidate groups include:
- D₆₆₈ (dihedral of order 1336)
- Z₄ × Z₁₆₇ 
- Z₆₆₈

The dihedral group D₄ₜ is known to produce many cocyclic Hadamard matrices. Álvarez et al. developed genetic algorithms specifically for cocyclic Hadamard search over dihedral groups, with classification results up to order 44.

## Key Challenge
Computing H²(D₆₆₈, Z₂) for a group of order 1336 is computationally intensive. The genetic algorithm must be adapted to handle the much larger search space at order 668.

## Implementation Backlog
- [ ] Compute H²(D₆₆₈, Z₂) using GAP cohomology package
- [ ] Enumerate representative cocycles
- [ ] Implement cocyclic matrix constructor
- [ ] Adapt Álvarez et al. guided-reproduction GA
- [ ] Profile orthogonality defect across cohomology classes
- [ ] Extend to Goethals-Seidel loops (Álvarez et al. 2019)
- [ ] Test with known cocyclic H(44) as validation
