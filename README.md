This repository contains examples of persistent Triton kernels and fusing persistent kernels with a global barrier.

It also contains the simplified code for two fused versions of the Mamba2 chunk state and state passing kernels:
- persistent global barrier fused
- concurrent fused with better performance

Run `main.py` to check correctness of the Mamba2 SSD using the original kernels vs the two partially fused versions.
