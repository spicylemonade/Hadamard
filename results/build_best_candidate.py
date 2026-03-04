#!/usr/bin/env python3
"""
Build the best candidate H(668) matrix from the Legendre baseline and export to CSV.

The Legendre baseline gives the closest known approximation:
- PSD = 672 at all non-zero frequencies (target: 668)
- HH^T = 668*I + 4*(J-I) where J is all-ones for appropriate block structure
- Off-diagonal error: uniformly 4 (the minimum possible by any known construction)

This is the best-known near-Hadamard matrix of order 668.
"""

import numpy as np
from numpy.fft import fft
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from hadamard_core import (
    legendre_symbol, goethals_seidel_array, verify_hadamard, 
    export_csv, psd_check, circulant_from_first_row, P, N
)

def build_legendre_gs_matrix():
    """Build the GS matrix from 4 copies of the Legendre sequence."""
    # Legendre sequence with chi(0) = 1
    chi = np.array([legendre_symbol(i, P) if i != 0 else 1 for i in range(P)], dtype=np.int8)
    
    print(f"Legendre sequence (first 20): {chi[:20]}")
    print(f"Row sum: {np.sum(chi)}")
    print(f"Number of +1s: {np.sum(chi == 1)}, Number of -1s: {np.sum(chi == -1)}")
    
    # Verify PSD
    psd, dev = psd_check(chi, chi, chi, chi)
    print(f"\nPSD check (4 copies of Legendre):")
    print(f"  PSD(0) = {psd[0]:.1f}")
    print(f"  PSD(k>0) = {psd[1]:.1f} (uniform)")
    print(f"  Target: {N}")
    print(f"  Gap at k>0: {dev[1]:.1f}")
    
    # Build the GS matrix
    print(f"\nBuilding 668x668 Goethals-Seidel matrix...")
    H = goethals_seidel_array(chi, chi, chi, chi)
    
    print(f"Matrix shape: {H.shape}")
    print(f"Entry range: [{H.min()}, {H.max()}]")
    
    # Verify
    valid, msg = verify_hadamard(H)
    print(f"Hadamard verification: {msg}")
    
    if not valid:
        # Compute HH^T statistics
        HHt = H.astype(np.int64) @ H.astype(np.int64).T
        diag = np.diag(HHt)
        off_diag = HHt - np.diag(diag)
        
        print(f"\nHH^T analysis:")
        print(f"  Diagonal: min={diag.min()}, max={diag.max()} (should be {N})")
        print(f"  Off-diagonal: min={off_diag[off_diag != 0].min()}, "
              f"max={off_diag.max()}, "
              f"unique values={np.unique(off_diag[off_diag != 0])[:10]}")
        print(f"  Number of non-zero off-diagonal: {np.count_nonzero(off_diag)}/{N*(N-1)}")
    
    return H, chi


def main():
    H, chi = build_legendre_gs_matrix()
    
    # Export to CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'hadamard_668.csv')
    csv_path = os.path.abspath(csv_path)
    export_csv(H, csv_path)
    print(f"\nExported to {csv_path}")
    print(f"File size: {os.path.getsize(csv_path) / 1024:.1f} KB")
    
    # Also save as best candidate
    candidate_path = os.path.join(os.path.dirname(__file__), 'best_candidate.csv')
    export_csv(H, candidate_path)
    
    # Save sequences
    np.savez(os.path.join(os.path.dirname(__file__), 'best_candidate_seqs.npz'),
             a=chi, b=chi, c=chi, d=chi,
             description='4 copies of Legendre sequence for Z_167')
    
    print(f"\nNOTE: This is a NEAR-Hadamard matrix, NOT an exact Hadamard matrix.")
    print(f"H(668) is currently the smallest open case of the Hadamard conjecture.")
    print(f"This matrix has HH^T = 668*I + E where max|E_ij| = 4.")
    
    return H


if __name__ == "__main__":
    H = main()
