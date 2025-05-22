# Copyright (c) 2024, Tri Dao, Albert Gu.

# Adapted from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/{...}.py

# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange

# for test data
from kernels.fused_concurrent import _fused_chunk_state_state_passing_fwd_concurrent
from kernels.fused_globalbar import _fused_chunk_state_state_passing_fwd_globalbar
from kernels.rand_data import generate_dummy_data

# import the original 5 main ssd kernels
from kernels.ssd_bmm import _bmm_chunk_fwd
from kernels.ssd_chunk_state import _chunk_cumsum_fwd
from kernels.ssd_chunk_state import _chunk_state_fwd
from kernels.ssd_state_passing import _state_passing_fwd
from kernels.ssd_chunk_scan import _chunk_scan_fwd


def dim_checks_common(x, dt, A, B, C, D=None, z=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()

    return x, dt, A, B, C, D, z, dstate


# runs the original kernels
def _mamba_chunk_scan_combined_fwd_original(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    x, dt, A, B, C, D, z, dstate = dim_checks_common(x, dt, A, B, C, D, z)

    dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=None, states_in_fp32=True)

    states, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                              initial_states=None,
                                              seq_idx=None, chunk_size=chunk_size, out_dtype=C.dtype)
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]

    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=None, output_dtype=torch.float32)
    out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=None)

    return out, out_x, dt, dA_cumsum, states, final_states


# runs with 2 kernels fused with a global barrier
def _mamba_chunk_scan_combined_fwd_globalbar(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    x, dt, A, B, C, D, z, dstate = dim_checks_common(x, dt, A, B, C, D, z)

    # fused, others original
    # (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, nheads, chunk_size, chunk_size)
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states, final_states = _fused_chunk_state_state_passing_fwd_globalbar(B, x, dt, dA_cumsum, C.dtype, initial_states=initial_states, seq_idx=seq_idx, states_in_fp32=True)
    out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx)

    return out, out_x, dt, dA_cumsum, states, final_states



# runs with 2 kernels fused concurrently
def _mamba_chunk_scan_combined_fwd_concurrent(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    x, dt, A, B, C, D, z, dstate = dim_checks_common(x, dt, A, B, C, D, z)

    # fused, others original
    # (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, nheads, chunk_size, chunk_size)
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states, final_states = _fused_chunk_state_state_passing_fwd_concurrent(B, x, dt, dA_cumsum, C.dtype, initial_states=initial_states, seq_idx=seq_idx, states_in_fp32=True)
    out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx)

    return out, out_x, dt, dA_cumsum, states, final_states





def check_same_tensors(t1, t2, tensor_name):
    if t1 is None and t2 is None:
        return True
    elif t1 is None:
        print(f"bad t1: {t1}")
    elif t2 is None:
        print(f"bad t2: {t2}")
    # retval = torch.allclose(t1, t2, rtol=0.01, atol=0.001, equal_nan=True)
    # if not retval:
    #     print(f"t1: {t1.reshape(-1)[:16]}, t2: {t2.reshape(-1)[:16]}")

    # return retval
    close = torch.isclose(t1, t2, rtol=0.01, atol=0.001, equal_nan=True)
    not_close_indices = torch.nonzero(~close, as_tuple=True)
    num_mismatch = len(not_close_indices[0])
    if num_mismatch > 0:
        print(f"num mismatch: {num_mismatch}")
        for i in range(min(5, num_mismatch)):
            # print(f"mismatch {i} at {[not_close_indices[x][] for x in }")
            indices = [not_close_indices[d][i].item() for d in range(len(not_close_indices))]
            print(f"mismatch {i} at index{indices} has {t1[*indices]} vs {t2[*indices]}")
            # print(f"mismatch {i}: {not_close_indices[i].tolist()} vs {not_close_indices[i].tolist()}")
    else:
        # print a few values from the middle
        flat1 = t1.reshape(-1).tolist()
        flat2 = t2.reshape(-1).tolist()
        elem_count = len(flat1)
        midpoint = elem_count // 2
        list1 = flat1[midpoint:midpoint+4]
        list2 = flat2[midpoint:midpoint+4]
        list1 = ["{:.4f}".format(item) for item in list1]
        list2 = ["{:.4f}".format(item) for item in list2]
        print(f"    {tensor_name} tensors match, example elems: t1: ...{list1}... vs t2: ...{list2}...")

    return num_mismatch == 0

def check_same_results(res1, res2):
    tuple_count = len(res1)
    print(f"    comparing {tuple_count} tensors")
    all_match = True
    tensor_names = ["out", "out_x", "dt", "dA_cumsum", "states", "final_states"]
    for i in range(tuple_count):
        this_match = check_same_tensors(res1[i], res2[i], tensor_names[i])
        all_match = all_match and this_match
        if not this_match:
            print(f"result {i} does not match")
    if all_match:
        print("results match")
    else:
        print("RESULTS DO NOT MATCH")

    return all_match



if __name__ == '__main__':
    print("hi")
    x, A, B, C, D, dt, dt_bias = generate_dummy_data()
    chunk_size = 256
    print("got random test tensors")
    original_res = _mamba_chunk_scan_combined_fwd_original(x, dt, A, B, C, chunk_size, D, dt_bias=dt_bias)
    print("ran original")
    print("--------------------checking...")
    original_match = check_same_results(original_res, original_res)
    print("--------------------checked original vs itself")
    globalbar_res = _mamba_chunk_scan_combined_fwd_globalbar(x, dt, A, B, C, chunk_size, D, dt_bias=dt_bias)
    print("ran globalbar")
    print("--------------------checking...")
    globalbar_match = check_same_results(globalbar_res, original_res)
    print("--------------------checked globalbar vs original")
    concurrent_res = _mamba_chunk_scan_combined_fwd_concurrent(x, dt, A, B, C, chunk_size, D, dt_bias=dt_bias)
    print("ran concurrent")
    print("--------------------checking...")
    concurrent_match = check_same_results(concurrent_res, original_res)
    print("--------------------checked concurrent vs original")
    print("----------------------------------------")
    print("----------------------------------------")
    print("results:", "all good" if original_match and globalbar_match and concurrent_match else "bad")
    print("original matches itself: ", original_match)
    print("global barrier fused matches original: ", globalbar_match)
    print("concurrent fused matches original: ", concurrent_match)
    print("----------------------------------------")
