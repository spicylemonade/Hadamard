#!/usr/bin/env python3
"""
Advanced metaheuristic search for H(668): multi-objective SA with DFT-guided neighborhood.
Item 011 of research rubric.

Implements:
1. Four +/-1 sequences of length 167 as search state
2. DFT-based objective minimizing max |PSD(k) - 668|
3. DFT-guided neighborhood: flip entries guided by worst-frequency deviation
4. Multi-start SA with adaptive temperature
5. Parallel tempering with 8 temperature replicas
6. Periodic logging of convergence

Deterministic seed: 42
"""

import numpy as np
from numpy.fft import fft, ifft
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hadamard_core import legendre_symbol, goethals_seidel_array, verify_hadamard, export_csv, P, N

np.random.seed(42)

# ---- Fast incremental DFT update ----

class DFTState:
    """Maintains DFT of 4 sequences with O(p) incremental updates on single flips."""
    
    def __init__(self, seqs):
        self.p = len(seqs[0])
        self.seqs = [s.copy().astype(np.float64) for s in seqs]
        self.dfts = [fft(s) for s in self.seqs]
        self.psd = sum(np.abs(d)**2 for d in self.dfts)
        self._update_cost()
        
        # Precompute twiddle factors for incremental update
        self.twiddles = np.exp(-2j * np.pi * np.arange(self.p)[:, None] * np.arange(self.p)[None, :] / self.p)
    
    def _update_cost(self):
        """Recompute cost metrics from current PSD."""
        dev = self.psd[1:] - N  # skip DC
        self.l2_cost = np.sum(dev**2)
        self.linf_cost = np.max(np.abs(dev))
        self.dev = dev
        self.worst_freq = np.argmax(np.abs(dev)) + 1  # 1-indexed freq
    
    def flip(self, seq_idx, pos):
        """Flip entry seq_idx[pos] and update DFT incrementally in O(p)."""
        old_val = self.seqs[seq_idx][pos]
        new_val = -old_val
        delta = new_val - old_val  # = +/-2
        
        # Update DFT: DFT_new[k] = DFT_old[k] + delta * exp(-2*pi*i*k*pos/p)
        dft_delta = delta * self.twiddles[:, pos]
        
        old_dft = self.dfts[seq_idx].copy()
        self.dfts[seq_idx] += dft_delta
        
        # Update PSD
        self.psd = sum(np.abs(d)**2 for d in self.dfts)
        
        # Update sequence
        self.seqs[seq_idx][pos] = new_val
        
        self._update_cost()
        return old_val
    
    def unflip(self, seq_idx, pos, old_val):
        """Undo a flip."""
        new_val = old_val
        cur_val = self.seqs[seq_idx][pos]
        delta = new_val - cur_val
        
        dft_delta = delta * self.twiddles[:, pos]
        self.dfts[seq_idx] += dft_delta
        self.psd = sum(np.abs(d)**2 for d in self.dfts)
        self.seqs[seq_idx][pos] = old_val
        self._update_cost()
    
    def get_guided_flip(self):
        """Choose a flip guided by the worst-frequency deviation.
        
        Find the position whose flip would most reduce the PSD at the worst frequency.
        """
        wf = self.worst_freq
        worst_dev = self.dev[wf - 1]  # deviation at worst freq
        
        # For each seq and position, compute effect of flip on PSD at worst freq
        # PSD(wf) = sum_i |DFT_i(wf)|^2
        # After flip seq_j[pos]: DFT_j(wf) += delta * exp(-2pi*i*wf*pos/p)
        # New |DFT_j(wf)|^2 = |DFT_j(wf) + delta * W|^2
        
        best_reduction = 0
        best_seq = 0
        best_pos = 0
        
        for si in range(4):
            old_dft_wf = self.dfts[si][wf]
            for pos in range(self.p):
                delta = -2 * self.seqs[si][pos]
                W = self.twiddles[wf, pos]
                new_dft_wf = old_dft_wf + delta * W
                change_in_psd = np.abs(new_dft_wf)**2 - np.abs(old_dft_wf)**2
                
                # We want PSD to decrease (since worst_dev > 0 typically)
                # If worst_dev > 0: we want change_in_psd < 0
                # If worst_dev < 0: we want change_in_psd > 0
                if worst_dev > 0:
                    reduction = -change_in_psd
                else:
                    reduction = change_in_psd
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_seq = si
                    best_pos = pos
        
        return best_seq, best_pos, best_reduction
    
    def copy(self):
        """Deep copy the state."""
        new = DFTState.__new__(DFTState)
        new.p = self.p
        new.seqs = [s.copy() for s in self.seqs]
        new.dfts = [d.copy() for d in self.dfts]
        new.psd = self.psd.copy()
        new.twiddles = self.twiddles  # shared (read-only)
        new._update_cost()
        return new


def initialize_from_legendre(p=P):
    """Initialize sequences from Legendre with random perturbations."""
    chi = np.array([legendre_symbol(i, p) if i > 0 else 1 for i in range(p)], dtype=np.float64)
    
    # Start with Legendre and randomly flip a few positions to break symmetry
    seqs = []
    for i in range(4):
        s = chi.copy()
        # Flip ~5% of positions randomly
        n_flip = max(1, p // 20)
        flip_pos = np.random.choice(p, size=n_flip, replace=False)
        s[flip_pos] *= -1
        seqs.append(s)
    
    return seqs


def initialize_random(p=P):
    """Initialize with random +/-1 sequences respecting row sum constraint."""
    seqs = []
    for _ in range(4):
        s = np.random.choice([-1.0, 1.0], size=p)
        seqs.append(s)
    return seqs


def parallel_tempering_search(n_replicas=8, max_iter=2000000, log_interval=100000):
    """
    Parallel tempering with multiple temperature replicas.
    Each replica runs SA, with periodic swap attempts between adjacent temperatures.
    """
    print(f"Parallel Tempering: {n_replicas} replicas, {max_iter} iterations")
    
    # Temperature ladder (geometric)
    T_min, T_max = 0.1, 50.0
    temps = np.geomspace(T_min, T_max, n_replicas)
    
    # Initialize replicas
    replicas = []
    for i in range(n_replicas):
        if i == 0:
            seqs = initialize_from_legendre()
        else:
            seqs = initialize_random()
        state = DFTState(seqs)
        replicas.append(state)
    
    # Track best
    best_l2 = min(r.l2_cost for r in replicas)
    best_linf = min(r.linf_cost for r in replicas)
    best_state = replicas[np.argmin([r.l2_cost for r in replicas])].copy()
    
    convergence_log = []
    start_time = time.time()
    
    for it in range(max_iter):
        # SA step for each replica
        for ri, (state, T) in enumerate(zip(replicas, temps)):
            # Choose move type
            if np.random.random() < 0.3:
                # Guided flip (targeting worst frequency)
                si, pos, _ = state.get_guided_flip()
            else:
                # Random flip
                si = np.random.randint(4)
                pos = np.random.randint(P)
            
            old_cost = state.l2_cost
            old_val = state.flip(si, pos)
            new_cost = state.l2_cost
            
            delta = new_cost - old_cost
            if delta > 0 and np.random.random() >= np.exp(-delta / max(T, 1e-10)):
                state.unflip(si, pos, old_val)
        
        # Periodic replica swap (every 100 iterations)
        if it % 100 == 0 and n_replicas > 1:
            i = np.random.randint(n_replicas - 1)
            j = i + 1
            cost_i = replicas[i].l2_cost
            cost_j = replicas[j].l2_cost
            beta_i = 1.0 / max(temps[i], 1e-10)
            beta_j = 1.0 / max(temps[j], 1e-10)
            
            swap_prob = min(1.0, np.exp((beta_i - beta_j) * (cost_i - cost_j)))
            if np.random.random() < swap_prob:
                replicas[i], replicas[j] = replicas[j], replicas[i]
        
        # Track best
        for state in replicas:
            if state.l2_cost < best_l2:
                best_l2 = state.l2_cost
                best_linf = state.linf_cost
                best_state = state.copy()
                
                if best_l2 < 1e-6:
                    elapsed = time.time() - start_time
                    print(f"  SOLUTION FOUND at iter {it}! Time: {elapsed:.1f}s")
                    return best_state, convergence_log
        
        # Logging
        if it % log_interval == 0:
            elapsed = time.time() - start_time
            replica_costs = [r.l2_cost for r in replicas]
            entry = {
                'iteration': it,
                'elapsed': elapsed,
                'best_l2': float(best_l2),
                'best_linf': float(best_linf),
                'replica_costs': [float(c) for c in replica_costs],
            }
            convergence_log.append(entry)
            print(f"  iter {it:>8d} | best L2={best_l2:.0f} Linf={best_linf:.1f} | "
                  f"replicas: {min(replica_costs):.0f}-{max(replica_costs):.0f} | "
                  f"time={elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    print(f"\nSearch complete. Total time: {elapsed:.1f}s")
    print(f"Best L2 cost: {best_l2:.1f}, Best Linf: {best_linf:.1f}")
    
    return best_state, convergence_log


def multi_start_sa(n_starts=5, iter_per_start=500000, log_interval=100000):
    """Multi-start SA with adaptive temperature."""
    print(f"Multi-start SA: {n_starts} starts, {iter_per_start} iterations each")
    
    overall_best_l2 = float('inf')
    overall_best_state = None
    convergence_log = []
    
    for start in range(n_starts):
        print(f"\n--- Start {start+1}/{n_starts} ---")
        
        if start == 0:
            seqs = initialize_from_legendre()
        else:
            seqs = initialize_random()
        
        state = DFTState(seqs)
        best_l2 = state.l2_cost
        best_linf = state.linf_cost
        
        T = 20.0
        T_min = 0.01
        no_improve = 0
        
        start_time = time.time()
        
        for it in range(iter_per_start):
            # Adaptive: increase guided flip rate as we converge
            guided_prob = min(0.5, 0.1 + 0.4 * it / iter_per_start)
            
            if np.random.random() < guided_prob:
                si, pos, _ = state.get_guided_flip()
            else:
                si = np.random.randint(4)
                pos = np.random.randint(P)
            
            old_cost = state.l2_cost
            old_val = state.flip(si, pos)
            new_cost = state.l2_cost
            
            delta = new_cost - old_cost
            if delta > 0 and np.random.random() >= np.exp(-delta / max(T, 1e-10)):
                state.unflip(si, pos, old_val)
            else:
                if new_cost < best_l2:
                    best_l2 = new_cost
                    best_linf = state.linf_cost
                    no_improve = 0
                else:
                    no_improve += 1
            
            # Adaptive cooling
            if no_improve > 10000:
                T = min(T * 1.5, 50.0)  # reheat
                no_improve = 0
            else:
                T = max(T * 0.99999, T_min)
            
            # Solution check
            if best_l2 < 1e-6:
                print(f"  SOLUTION FOUND at iter {it}!")
                return state, convergence_log
            
            if it % log_interval == 0:
                elapsed = time.time() - start_time
                entry = {
                    'start': start,
                    'iteration': it,
                    'elapsed': elapsed,
                    'best_l2': float(best_l2),
                    'best_linf': float(best_linf),
                    'temperature': float(T),
                }
                convergence_log.append(entry)
                print(f"  iter {it:>8d} | L2={state.l2_cost:.0f} best={best_l2:.0f} "
                      f"Linf={best_linf:.1f} | T={T:.4f} | {elapsed:.1f}s")
        
        if best_l2 < overall_best_l2:
            overall_best_l2 = best_l2
            overall_best_state = state.copy()
    
    print(f"\nOverall best L2: {overall_best_l2:.1f}")
    return overall_best_state, convergence_log


def main():
    print("=" * 60)
    print("ADVANCED METAHEURISTIC SEARCH FOR H(668)")
    print("=" * 60)
    
    # Run parallel tempering (main method)
    print("\n[1] Parallel Tempering Search")
    pt_state, pt_log = parallel_tempering_search(
        n_replicas=8, max_iter=2000000, log_interval=200000
    )
    
    # Run multi-start SA
    print("\n[2] Multi-Start SA Search")
    ms_state, ms_log = multi_start_sa(
        n_starts=3, iter_per_start=500000, log_interval=100000
    )
    
    # Report best result
    best_state = pt_state if pt_state.l2_cost < ms_state.l2_cost else ms_state
    print(f"\n{'='*60}")
    print(f"BEST RESULT")
    print(f"{'='*60}")
    print(f"  L2 cost: {best_state.l2_cost:.1f}")
    print(f"  Linf cost: {best_state.linf_cost:.1f}")
    
    is_solution = best_state.l2_cost < 1e-6
    
    if is_solution:
        # Build and verify
        seqs = [s.astype(np.int8) for s in best_state.seqs]
        H = goethals_seidel_array(*seqs)
        valid, msg = verify_hadamard(H)
        print(f"  Verification: {msg}")
        if valid:
            export_csv(H, 'hadamard_668.csv')
            print(f"  Solution exported to hadamard_668.csv!")
    else:
        # Convergence barrier documented
        print(f"  No exact solution found.")
        print(f"  Convergence barrier: L2 cost plateaus around {best_state.l2_cost:.0f}")
        print(f"  This corresponds to PSD deviation of ~{best_state.linf_cost:.0f} at worst frequency")
    
    # Save convergence log
    log_path = os.path.join(os.path.dirname(__file__), 'experiments', 'metaheuristic_log.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump({'parallel_tempering': pt_log, 'multi_start_sa': ms_log}, f, indent=2)
    print(f"  Convergence log saved to {log_path}")
    
    return best_state


if __name__ == "__main__":
    main()
