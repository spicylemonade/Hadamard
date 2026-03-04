# RL-Guided Difference Family Construction for H(668)

## Context
Deep reinforcement learning has demonstrated remarkable ability to discover strategies in combinatorial domains — from Go (AlphaGo) to protein folding (AlphaFold). The SDS construction problem has a natural sequential structure: elements are added one at a time to four blocks, creating an MDP amenable to RL training.

The Cayley graph of Z_167 provides a natural graph structure for a graph neural network (GNN) policy. The algebraic structure of the group (automorphisms, subgroup lattice) should be discoverable by the GNN through training.

## Key Innovation
Curriculum learning enables transfer from small known instances (v=11, 23, 47, 59, 83) to the target v=167. The GNN policy is inherently transferable across different group sizes because it operates on local graph neighborhoods.

## Implementation Backlog
- [ ] Define MDP for SDS construction over Z_v
- [ ] Implement Cayley graph construction for Z_v
- [ ] Build GNN policy network (message-passing architecture)
- [ ] Implement PPO training loop
- [ ] Train and validate on v=11 (order 44, known SDS)
- [ ] Transfer to v=23, v=47 (known SDS)
- [ ] Analyze learned policies (do they discover cyclotomic structure?)
- [ ] Scale to v=167 with partial solution harvesting
- [ ] Integrate with classical refinement (GA or SA) for completion
