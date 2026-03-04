#!/usr/bin/env python3
"""
Intensive search for Hadamard matrix H(668) via Goethals-Seidel array.

The approach:
1. Start from Legendre sequences perturbed to have correct row sums (PSD(0)=668)
2. Use high-performance SA with row-sum-preserving swaps
3. Multiple restarts with different perturbation patterns
4. Exploit Z_167 symmetry and frequency-targeting

Key constraint: 4 ±1 sequences of length 167 with
  PSD(k) = sum |DFT_i(k)|^2 = 668 for ALL k=0,...,166
  where PSD(0) = s1^2+s2^2+s3^2+s4^2 = 668 fixes row sums
"""

import numpy as np
from numpy.fft import fft, ifft
from numba import njit
import time
import sys
import os

P = 167
N = 668

# Valid row sum quadruples (s1^2+s2^2+s3^2+s4^2 = 668, all odd)
QUADS = [
    (7, 13, 15, 15),
    (3, 9, 17, 17),
    (3, 3, 17, 19),
    (5, 9, 11, 21),
    (1, 9, 15, 19),
    (3, 7, 13, 21),
    (1, 1, 15, 21),
    (3, 3, 11, 23),
    (3, 7, 9, 23),
    (3, 3, 5, 25),
]


def legendre_seq():
    seq = np.zeros(P, dtype=np.int8)
    for a in range(P):
        if a == 0:
            seq[0] = 1
        else:
            val = pow(a, (P - 1) // 2, P)
            seq[a] = 1 if val == 1 else -1
    return seq


def perturb_to_rowsum(seq, target_sum, rng):
    """Perturb a ±1 sequence to achieve a target row sum."""
    s = seq.copy()
    current = int(np.sum(s))
    diff = target_sum - current
    if diff > 0:
        neg_pos = np.where(s == -1)[0]
        flip_count = diff // 2
        if flip_count > len(neg_pos):
            return None
        idx = rng.choice(neg_pos, flip_count, replace=False)
        s[idx] = 1
    elif diff < 0:
        pos_pos = np.where(s == 1)[0]
        flip_count = (-diff) // 2
        if flip_count > len(pos_pos):
            return None
        idx = rng.choice(pos_pos, flip_count, replace=False)
        s[idx] = -1
    return s


@njit(cache=True)
def compute_dft_components(seqs):
    """Compute DFT real/imag for all 4 sequences."""
    dft_re = np.zeros((4, P))
    dft_im = np.zeros((4, P))
    omega = 2.0 * np.pi / P
    for s in range(4):
        for k in range(P):
            re, im = 0.0, 0.0
            for j in range(P):
                angle = omega * k * j
                re += seqs[s, j] * np.cos(angle)
                im -= seqs[s, j] * np.sin(angle)
            dft_re[s, k] = re
            dft_im[s, k] = im
    return dft_re, dft_im


@njit(cache=True)
def psd_from_dft(dft_re, dft_im):
    """Compute PSD from DFT components."""
    psd = np.zeros(P)
    for k in range(P):
        for s in range(4):
            psd[k] += dft_re[s, k]**2 + dft_im[s, k]**2
    return psd


@njit(cache=True)
def l2_cost(psd):
    cost = 0.0
    for k in range(P):
        d = psd[k] - N
        cost += d * d
    return cost


@njit(cache=True)
def linf_cost(psd):
    mx = 0.0
    for k in range(P):
        d = abs(psd[k] - N)
        if d > mx:
            mx = d
    return mx


@njit(cache=True)
def update_dft_swap(dft_re, dft_im, psd, seq_idx, pos1, pos2):
    """
    Update DFT and PSD after swapping values at pos1 and pos2 in sequence seq_idx.
    Both positions must have different signs. This preserves row sum.
    """
    # Value at pos1 becomes -old, and same for pos2
    omega = 2.0 * np.pi / P
    for k in range(P):
        old_pow = dft_re[seq_idx, k]**2 + dft_im[seq_idx, k]**2
        
        angle1 = omega * k * pos1
        angle2 = omega * k * pos2
        # pos1: flip from +1 to -1 (diff = -2) or -1 to +1 (diff = +2)
        # pos2: opposite flip
        # Actually we swap: seq[pos1] ↔ seq[pos2]
        # But since they have different signs, this is equivalent to flipping both.
        # diff1 = seq[pos2] - seq[pos1] = -2*seq[pos1]
        # diff2 = seq[pos1] - seq[pos2] = -2*seq[pos2] = 2*seq[pos1]
        # Hmm, this is swap, not independent flips.
        
        # For a swap of positions pos1 and pos2 where seq[pos1] = v1, seq[pos2] = v2:
        # DFT change: sum_{j} seq'[j] * e^{-i*omega*k*j} - sum_{j} seq[j] * e^{-i*omega*k*j}
        # = (v2-v1)*(cos(omega*k*pos1) - i*sin(omega*k*pos1)) + (v1-v2)*(cos(omega*k*pos2) - i*sin(omega*k*pos2))
        # = (v2-v1)*[cos(a1)-cos(a2)] + i*(v2-v1)*[-sin(a1)+sin(a2)]
        # Wait, I need to be more careful. DFT convention: X[k] = sum x[j] * e^{-i*2pi*k*j/N}
        # So re(X[k]) = sum x[j]*cos(2pi*k*j/N), im(X[k]) = -sum x[j]*sin(2pi*k*j/N)
        
        # diff at pos1: (v2-v1)*cos(a1) for real, -(v2-v1)*sin(a1) for imag
        # diff at pos2: (v1-v2)*cos(a2) for real, -(v1-v2)*sin(a2) for imag
        # Total: (v2-v1)*(cos(a1)-cos(a2)) for real, (v2-v1)*(sin(a2)-sin(a1)) for imag
        # But we track im as: dft_im = -sum x[j]*sin(...), so change in dft_im:
        # -(v2-v1)*sin(a1) - (-(v1-v2)*sin(a2)) = -(v2-v1)*sin(a1) + (v1-v2)*sin(a2)
        # = (v1-v2)*(sin(a1)+sin(a2))... no wait.
        # delta_im = -(v2-v1)*sin(a1) + (v1-v2)*sin(a2) ... hmm I'm confusing myself.
        
        # Let me just do it step by step.
        # Before swap: seq[pos1] = v1, seq[pos2] = v2
        # After swap: seq[pos1] = v2, seq[pos2] = v1
        # Change at pos1: diff1 = v2 - v1
        # Change at pos2: diff2 = v1 - v2 = -diff1
        
        cos1 = np.cos(angle1)
        sin1 = np.sin(angle1)
        cos2 = np.cos(angle2)
        sin2 = np.sin(angle2)
        
        # Since we stored things differently, let me just use the standard approach:
        # dft_re += diff1*cos1 + diff2*cos2 = diff1*(cos1 - cos2)
        # dft_im -= diff1*sin1 + diff2*sin2 = -(diff1*(sin1 - sin2))
        # But diff1 and diff2 = -diff1... 
        # I'll just handle this properly below.
        pass
    
    # OK let me just do this the clean way
    return psd  # placeholder


@njit(cache=True)
def sa_rowsum_preserving(seqs_init, max_iters, T_start, T_end, seed):
    """
    SA search with row-sum-preserving moves.
    Each move: pick a sequence, swap a +1 position with a -1 position.
    """
    np.random.seed(seed)
    seqs = seqs_init.copy().astype(np.float64)
    
    # Compute DFTs
    omega = 2.0 * np.pi / P
    # Precompute cos/sin table
    cos_table = np.zeros((P, P))
    sin_table = np.zeros((P, P))
    for k in range(P):
        for j in range(P):
            angle = omega * k * j
            cos_table[k, j] = np.cos(angle)
            sin_table[k, j] = np.sin(angle)
    
    dft_re = np.zeros((4, P))
    dft_im = np.zeros((4, P))
    for s in range(4):
        for k in range(P):
            re, im = 0.0, 0.0
            for j in range(P):
                re += seqs[s, j] * cos_table[k, j]
                im -= seqs[s, j] * sin_table[k, j]
            dft_re[s, k] = re
            dft_im[s, k] = im
    
    psd = np.zeros(P)
    for k in range(P):
        for s in range(4):
            psd[k] += dft_re[s, k]**2 + dft_im[s, k]**2
    
    current_cost = l2_cost(psd)
    best_cost = current_cost
    best_linf = linf_cost(psd)
    best_seqs = seqs.copy()
    
    # Precompute positions of +1 and -1 for each sequence
    # (updated as we go)
    pos_plus = [[0]*P for _ in range(4)]  # will resize
    pos_minus = [[0]*P for _ in range(4)]
    n_plus = np.zeros(4, dtype=np.int64)
    n_minus = np.zeros(4, dtype=np.int64)
    
    for s in range(4):
        pp, pm = 0, 0
        for j in range(P):
            if seqs[s, j] > 0:
                pos_plus[s][pp] = j
                pp += 1
            else:
                pos_minus[s][pm] = j
                pm += 1
        n_plus[s] = pp
        n_minus[s] = pm
    
    cooling = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for it in range(max_iters):
        # Pick random sequence
        s = np.random.randint(4)
        
        # Pick random +1 position and random -1 position
        if n_plus[s] == 0 or n_minus[s] == 0:
            continue
        p1_idx = np.random.randint(n_plus[s])
        p2_idx = np.random.randint(n_minus[s])
        pos1 = pos_plus[s][p1_idx]
        pos2 = pos_minus[s][p2_idx]
        
        # Compute cost change from swapping pos1 (+1 → -1) and pos2 (-1 → +1)
        # diff1 = -2 (pos1 goes from +1 to -1)
        # diff2 = +2 (pos2 goes from -1 to +1)
        
        delta_cost = 0.0
        # Save old DFT values to revert if needed
        old_re = np.zeros(P)
        old_im = np.zeros(P)
        old_psd = psd.copy()
        
        for k in range(P):
            old_re[k] = dft_re[s, k]
            old_im[k] = dft_im[s, k]
            
            old_pow = dft_re[s, k]**2 + dft_im[s, k]**2
            
            # Update DFT
            dft_re[s, k] += -2.0 * cos_table[k, pos1] + 2.0 * cos_table[k, pos2]
            dft_im[s, k] -= -2.0 * sin_table[k, pos1] + 2.0 * sin_table[k, pos2]
            
            new_pow = dft_re[s, k]**2 + dft_im[s, k]**2
            psd[k] += new_pow - old_pow
        
        new_cost = l2_cost(psd)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-30)):
            # Accept
            current_cost = new_cost
            seqs[s, pos1] = -1.0
            seqs[s, pos2] = 1.0
            
            # Update position lists
            pos_plus[s][p1_idx] = pos2  # pos2 is now +1
            pos_minus[s][p2_idx] = pos1  # pos1 is now -1
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_linf = linf_cost(psd)
                for ss in range(4):
                    for jj in range(P):
                        best_seqs[ss, jj] = seqs[ss, jj]
        else:
            # Revert DFT and PSD
            for k in range(P):
                dft_re[s, k] = old_re[k]
                dft_im[s, k] = old_im[k]
            for k in range(P):
                psd[k] = old_psd[k]
        
        T *= cooling
    
    return best_seqs, best_cost, best_linf


@njit(cache=True)
def sa_free_flip(seqs_init, max_iters, T_start, T_end, seed):
    """
    SA search with free single flips (no row sum constraint).
    Allows the search to explore more of the landscape.
    """
    np.random.seed(seed)
    seqs = seqs_init.copy().astype(np.float64)
    
    omega = 2.0 * np.pi / P
    cos_table = np.zeros((P, P))
    sin_table = np.zeros((P, P))
    for k in range(P):
        for j in range(P):
            angle = omega * k * j
            cos_table[k, j] = np.cos(angle)
            sin_table[k, j] = np.sin(angle)
    
    dft_re = np.zeros((4, P))
    dft_im = np.zeros((4, P))
    for s in range(4):
        for k in range(P):
            re, im = 0.0, 0.0
            for j in range(P):
                re += seqs[s, j] * cos_table[k, j]
                im -= seqs[s, j] * sin_table[k, j]
            dft_re[s, k] = re
            dft_im[s, k] = im
    
    psd = np.zeros(P)
    for k in range(P):
        for s in range(4):
            psd[k] += dft_re[s, k]**2 + dft_im[s, k]**2
    
    current_cost = l2_cost(psd)
    best_cost = current_cost
    best_linf = linf_cost(psd)
    best_seqs = seqs.copy()
    
    cooling = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for it in range(max_iters):
        s = np.random.randint(4)
        pos = np.random.randint(P)
        
        old_val = seqs[s, pos]
        diff = -2.0 * old_val  # flip: +1→-1 gives diff=-2, -1→+1 gives diff=+2
        
        old_psd = psd.copy()
        old_re_vals = np.zeros(P)
        old_im_vals = np.zeros(P)
        
        for k in range(P):
            old_re_vals[k] = dft_re[s, k]
            old_im_vals[k] = dft_im[s, k]
            old_pow = dft_re[s, k]**2 + dft_im[s, k]**2
            dft_re[s, k] += diff * cos_table[k, pos]
            dft_im[s, k] -= diff * sin_table[k, pos]
            new_pow = dft_re[s, k]**2 + dft_im[s, k]**2
            psd[k] += new_pow - old_pow
        
        new_cost = l2_cost(psd)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-30)):
            current_cost = new_cost
            seqs[s, pos] = -old_val
            if new_cost < best_cost:
                best_cost = new_cost
                best_linf = linf_cost(psd)
                for ss in range(4):
                    for jj in range(P):
                        best_seqs[ss, jj] = seqs[ss, jj]
        else:
            for k in range(P):
                dft_re[s, k] = old_re_vals[k]
                dft_im[s, k] = old_im_vals[k]
                psd[k] = old_psd[k]
        
        T *= cooling
    
    return best_seqs, best_cost, best_linf


def run_campaign(total_time=600, seed=42, verbose=True):
    """Run a comprehensive search campaign."""
    leg = legendre_seq()
    rng = np.random.RandomState(seed)
    
    best_cost = float('inf')
    best_linf = float('inf')
    best_seqs = None
    
    start = time.time()
    run = 0
    
    if verbose:
        print(f"Starting H(668) search campaign. Time budget: {total_time}s")
        print(f"="*70)
    
    while time.time() - start < total_time:
        run += 1
        elapsed = time.time() - start
        remaining = total_time - elapsed
        if remaining < 10:
            break
        
        # Cycle through quadruples and strategies
        quad_idx = run % len(QUADS)
        quad = QUADS[quad_idx]
        strategy = (run // len(QUADS)) % 3
        
        run_seed = seed + run * 137
        local_rng = np.random.RandomState(run_seed)
        
        # Initialize from Legendre + perturbation
        seqs = np.zeros((4, P), dtype=np.float64)
        sums = list(quad)
        # Randomly permute which sum goes to which sequence
        local_rng.shuffle(sums)
        # Also randomly assign signs
        for i in range(4):
            if local_rng.random() < 0.5:
                sums[i] = -sums[i]
        
        for i in range(4):
            s = perturb_to_rowsum(leg.copy(), sums[i], local_rng)
            if s is not None:
                seqs[i] = s.astype(np.float64)
            else:
                seqs[i] = leg.copy().astype(np.float64)
        
        # Check PSD
        psd = np.zeros(P)
        for s_idx in range(4):
            sf = fft(seqs[s_idx])
            psd += np.abs(sf)**2
        init_cost = np.sum((psd - N)**2)
        
        # Choose iterations based on remaining time
        n_iters = min(5_000_000, max(500_000, int(remaining * 100_000)))
        
        if strategy == 0:
            # Row-sum preserving SA
            T_start, T_end = 50.0, 0.001
            result_seqs, result_cost, result_linf = sa_rowsum_preserving(
                seqs, n_iters, T_start, T_end, run_seed
            )
            strat_name = "RS-SA"
        elif strategy == 1:
            # Free flip SA (high temp start)
            T_start, T_end = 200.0, 0.0001
            result_seqs, result_cost, result_linf = sa_free_flip(
                seqs, n_iters, T_start, T_end, run_seed
            )
            strat_name = "Free-SA"
        else:
            # Two-phase: free SA then row-sum SA
            n1 = n_iters // 2
            n2 = n_iters - n1
            temp_seqs, temp_cost, temp_linf = sa_free_flip(
                seqs, n1, 100.0, 1.0, run_seed
            )
            result_seqs, result_cost, result_linf = sa_rowsum_preserving(
                temp_seqs, n2, 10.0, 0.001, run_seed + 1
            )
            strat_name = "2Phase"
        
        run_time = time.time() - start - elapsed
        
        if verbose:
            actual_sums = [int(np.sum(result_seqs[i])) for i in range(4)]
            psd_0 = sum(s**2 for s in actual_sums)
            print(f"Run {run:3d} [{strat_name:6s}] quad={quad} iters={n_iters/1e6:.1f}M "
                  f"L2={result_cost:.0f} Linf={result_linf:.1f} "
                  f"sums={actual_sums} PSD0={psd_0} [{run_time:.1f}s]")
        
        if result_cost < best_cost:
            best_cost = result_cost
            best_linf = result_linf
            best_seqs = result_seqs.copy()
            if verbose:
                print(f"  *** NEW BEST: L2={best_cost:.0f}, Linf={best_linf:.1f}")
            
            if best_cost < 1.0:
                if verbose:
                    print("*** EXACT SOLUTION FOUND! ***")
                break
    
    if verbose:
        total = time.time() - start
        print(f"\n{'='*70}")
        print(f"Campaign complete in {total:.1f}s over {run} runs")
        print(f"Best L2={best_cost:.0f}, Linf={best_linf:.1f}")
        if best_seqs is not None:
            sums = [int(np.sum(best_seqs[i])) for i in range(4)]
            print(f"Best row sums: {sums}")
    
    return best_seqs, best_cost, best_linf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    seqs, cost, linf = run_campaign(args.time, args.seed)
    
    if cost < 1.0:
        print("\nVerifying exact solution...")
        seqs_int = np.round(seqs).astype(np.int8)
        
        # Build GS matrix
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from hadamard_core import goethals_seidel_array, verify_hadamard, export_csv
        
        H = goethals_seidel_array(seqs_int[0], seqs_int[1], seqs_int[2], seqs_int[3])
        valid, msg = verify_hadamard(H)
        print(f"Verification: {msg}")
        
        if valid:
            export_csv(H, os.path.join(os.path.dirname(__file__), '..', 'hadamard_668.csv'))
            print("SAVED hadamard_668.csv")
            np.savez(os.path.join(os.path.dirname(__file__), 'solution_sequences.npz'),
                     a=seqs_int[0], b=seqs_int[1], c=seqs_int[2], d=seqs_int[3])
    
    # Save best result
    np.savez(os.path.join(os.path.dirname(__file__), 'intensive_campaign_best.npz'),
             seqs=np.round(seqs).astype(np.int8) if seqs is not None else np.array([]),
             cost=cost, linf=linf)
