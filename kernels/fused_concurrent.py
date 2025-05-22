
# TODO: fix autotune configs to be good
# this is just a basic combo of the original chunk_state and state_passing configs
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # WARNING: can deadlock if SKIP_THREADBLOCK_COUNT is too high!!!
        triton.Config({'BLOCK_SIZE_M': 64//2,  'BLOCK_SIZE_N': 128//2, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_SP': 1024, 'SKIP_THREADBLOCK_COUNT': 16}, num_stages=1, num_warps=8),
    ],
    key=['hdim', 'dstate', 'chunk_size', 'dim'],
)
@triton.jit
def _fused_chunk_state_state_passing_fwd_kernel_concurrent(
    grid_atomic,
    grid_atomic2,
    chunk_work_done,
    # grid_size_SMs,
    min_concurrent_threadblocks,

    # Pointers to matrices
    x_ptr, b_ptr, states_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nheads_ngroups_ratio, nheads, nchunks,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate, stride_states_dim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    


    # Pointers to matrices
    out_ptr, final_states_ptr, dA_cs_ptr, initstates_ptr,
    # Strides
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_final_states_batch, stride_final_states_head, stride_final_states_dim,
    offset_dA_cs_last_elem,
    stride_initstates_batch, stride_initstates_head, stride_initstates_dim,

    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_SP: tl.constexpr, SKIP_THREADBLOCK_COUNT: tl.constexpr,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
):
    grid_dim_headdim = tl.cdiv(hdim, BLOCK_SIZE_M)
    grid_dim_dstate = tl.cdiv(dstate, BLOCK_SIZE_N)
    target_grid_size = batch * nchunks * nheads * grid_dim_headdim * grid_dim_dstate

    b_ptr_base = b_ptr
    x_ptr_base = x_ptr
    dt_ptr_base = dt_ptr
    dA_cumsum_ptr_base = dA_cumsum_ptr
    if HAS_SEQ_IDX:
        seq_idx_ptr_base = seq_idx_ptr
    states_ptr_base = states_ptr

    # TODO: set a good number or add to config, the number of threadblocks to skip to step 2
    # must be less than the number of concurrent threadblocks to avoid deadlocks
    should_do_step1 = tl.program_id(0) >= SKIP_THREADBLOCK_COUNT# < (grid_size_SMs * 15/16)
    if should_do_step1:
        # my_block_id = tl.atomic_add(grid_atomic, 1).to(tl.int32)

        # while(my_block_id < target_grid_size):
            my_block_id = tl.program_id(0) - SKIP_THREADBLOCK_COUNT

            pid_c = my_block_id % nchunks
            pid_n = (my_block_id // nchunks) % grid_dim_dstate
            pid_m = (my_block_id // (nchunks * grid_dim_dstate)) % grid_dim_headdim
            pid_h = (my_block_id // (nchunks * grid_dim_dstate * grid_dim_headdim)) % nheads
            pid_b = (my_block_id // (nchunks * grid_dim_dstate * grid_dim_headdim * nheads)) % batch

            # wait for less than 4 work
            max_work = hdim * dstate * nchunks * 4
            cur_work = tl.atomic_add(grid_atomic, 0)
            while cur_work >= max_work: # spin till sem below max_work
                cur_work = tl.atomic_add(grid_atomic, 0)


            b_ptr   = b_ptr_base + pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
            x_ptr   = x_ptr_base + pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
            dt_ptr  = dt_ptr_base + pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
            dA_cumsum_ptr = dA_cumsum_ptr_base + pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
            if HAS_SEQ_IDX:
                seq_idx_ptr = seq_idx_ptr_base + pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
            b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
            dt_ptrs = dt_ptr + offs_k * stride_dt_csize
            dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32) # TODO: consider using float16
            dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
            if HAS_SEQ_IDX:
                seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

            chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
            if HAS_SEQ_IDX:
                seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
            # for k in tl.range(0, chunk_size_limit, BLOCK_SIZE_K, num_stages=4):
                x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0, cache_modifier=".cv")
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0, cache_modifier=".cv").to(tl.float32)
                dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
                if HAS_SEQ_IDX:
                    seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)
                dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
                if not HAS_SEQ_IDX:
                    scale = tl.exp((dA_cs_last - dA_cs_k)) * dt_k
                else:
                    scale = tl.where(seq_idx_k == seq_idx_last, tl.exp((dA_cs_last - dA_cs_k)) * dt_k, 0.0)
                b *= scale[:, None]
                b = b.to(x_ptr.dtype.element_ty)
                acc += tl.dot(x, b)
                x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
                b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
                dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
                dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
                if HAS_SEQ_IDX:
                    seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
            states = acc.to(states_ptr.dtype.element_ty)

            states_ptr = states_ptr_base + pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
            c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
            tl.store(states_ptrs, states, mask=c_mask, cache_modifier=".cg")

            # my_block_id = tl.atomic_add(grid_atomic, 1).to(tl.int32)
            tl.atomic_add(chunk_work_done + pid_b * nheads + pid_h, 1) # mark work as done
            tl.atomic_add(grid_atomic, BLOCK_SIZE_M * BLOCK_SIZE_N) # sem down

    
    # # busy wait for global barrier, this is ok since we're in a persistent kernel
    # # we can check that not only have all block started (grid_atomic == target_grid_size),
    # # but that all blocks have finished and further incremented the grid_atomic counter
    # # i.e. grid_atomic == target_grid_size + grid_size_SMs

    # old_grid_atomic = tl.atomic_add(grid_atomic, 0).to(tl.int32) # atomic load TODO: check if there's a better atomic load
    # while(old_grid_atomic < target_grid_size + grid_size_SMs):
    #     old_grid_atomic = tl.atomic_add(grid_atomic, 0).to(tl.int32)

    # Note:
    # Step 1 processes a block of size BLOCK_SIZE_M * BLOCK_SIZE_N in the hdim * dstate dimensions, with pid m, n, c, b, h
    # Step 2 processes a block of size BLOCK_SIZE_SP in the dim dimension, with pid m2, b, h
    # if we want to overlap them, we want Step 1 to do work in the order than Step 2 consumes the results
    # in order to start work, we need a whole m2 block to start
    # in order to keep having stuff to do, both kernels must process the other dims in the same order
    # therefore, we can have b, h, and even m2 (if BLOCK_SIZE_SP = BLOCK_SIZE_M * BLOCK_SIZE_N) in the same order as batch dimensions
    # but Step 1 must have finished all of the batch along the c dimension for the work to be ready (since Step 2 iterates all chunks)

    # # we use this combined dim, but restrict the block size to be the same as step 1
    # BLOCK_SIZE_SP = BLOCK_SIZE_M * BLOCK_SIZE_N

    # must either be a skip or one of the last concurrent blocks to avoid deadlock
    should_do_step2 = not should_do_step1 # or (min_concurrent_threadblocks - tl.program_id(0)) < tl.num_programs(0) 
    if should_do_step2:
        dim = hdim * dstate
        grid_dim_dim = tl.cdiv(dim, BLOCK_SIZE_SP)
        target_grid_size2 = grid_dim_dim * batch * nheads

        dA_cs_ptr_base = dA_cs_ptr + offset_dA_cs_last_elem
        out_ptr_base = out_ptr
        final_states_ptr_base = final_states_ptr
        if HAS_INITSTATES:
            initstates_ptr_base = initstates_ptr
        if HAS_SEQ_IDX:
            seq_idx_ptr_base = seq_idx_ptr

        my_block_id = tl.atomic_add(grid_atomic2, 1).to(tl.int32)

        while(my_block_id < target_grid_size2):
        # if my_block_id < target_grid_size2:

            # pid_h = my_block_id % nheads
            # pid_b = (my_block_id // nheads) % batch
            # pid_dim = (my_block_id // (nheads * batch)) # % grid_dim_dim # don't need last mod
            # # pid_n = (my_block_id // (nheads * batch)) % grid_dim_dstate
            # # pid_m = (my_block_id // (nheads * batch * grid_dim_dstate)) # % grid_dim_headdim # don't need last mod
            pid_dim = my_block_id % grid_dim_dim
            pid_h = (my_block_id // grid_dim_dim) % nheads
            pid_b = (my_block_id // (grid_dim_dim * nheads)) % batch


            # instead of the global barrier, we only need to check that all chunks were processed in step 1.
            # just check chunk_work_done batch, nheads, nchunks along the c dimension
            # really sus-looking spin wait
            # work_done = tl.atomic_add(chunk_work_done + pid_b * nheads * nchunks + pid_h * nchunks + tl.arange(0, nchunks), 0)
            # # check if any too small
            # all_work_done = tl.min(work_done) == 1 # TODO: doesn't seem to work in the condition of the for loop directly??
            # while (not all_work_done):
            #     work_done = tl.atomic_add(chunk_work_done + pid_b * nheads * nchunks + pid_h * nchunks + tl.arange(0, nchunks), 0)#, mask=work_done < 1)
            #     all_work_done = tl.min(work_done) == 1
            work_done = tl.atomic_add(chunk_work_done + pid_b * nheads + pid_h, 0)
            work_needed = nchunks * grid_dim_headdim * grid_dim_dstate # TODO: don't bunch up by these 2 grid dim
            while (work_done < work_needed):
                work_done = tl.atomic_add(chunk_work_done + pid_b * nheads + pid_h, 0)


            states_ptr = states_ptr_base + pid_b * stride_states_batch + pid_h * stride_states_head
            dA_cs_ptr  = dA_cs_ptr_base + pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
            out_ptr    = out_ptr_base + pid_b * stride_out_batch + pid_h * stride_out_head
            final_states_ptr = final_states_ptr_base + pid_b * stride_final_states_batch + pid_h * stride_final_states_head
            if HAS_INITSTATES:
                initstates_ptr = initstates_ptr_base + pid_b * stride_initstates_batch + pid_h * stride_initstates_head
            if HAS_SEQ_IDX:
                seq_idx_ptr = seq_idx_ptr_base + pid_b * stride_seq_idx_batch

            offs_dim = pid_dim * BLOCK_SIZE_SP + tl.arange(0, BLOCK_SIZE_SP)
            states_ptrs = states_ptr + offs_dim * stride_states_dim
            out_ptrs = out_ptr + offs_dim * stride_out_dim
            final_states_ptrs = final_states_ptr + offs_dim * stride_final_states_dim

            if not HAS_INITSTATES:
                states = tl.zeros((BLOCK_SIZE_SP, ), dtype=tl.float32)
            else:
                initstates_ptrs = initstates_ptr + offs_dim * stride_initstates_dim
                states = tl.load(initstates_ptrs, mask=offs_dim < dim, other=0.0).to(tl.float32)
            tl.store(out_ptrs, states, mask=offs_dim < dim)
            out_ptrs += stride_out_chunk
            seq_idx = 0
            for c in range(nchunks):
            # for c in tl.range(nchunks, num_stages=2):
                new_states = tl.load(states_ptrs, mask=offs_dim < dim, other=0.0, cache_modifier=".cv").to(tl.float32)
                dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
                scale = tl.exp(dA_cs)
                if HAS_SEQ_IDX:
                    seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
                    scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
                    seq_idx = seq_idx_new
                states = scale * states + new_states
                if c < nchunks - 1:
                    tl.store(out_ptrs, states, mask=offs_dim < dim, cache_modifier=".wt")
                else:
                    tl.store(final_states_ptrs, states, mask=offs_dim < dim, cache_modifier=".wt")
                states_ptrs += stride_states_chunk
                dA_cs_ptr += stride_dA_cs_chunk
                out_ptrs += stride_out_chunk

            my_block_id = tl.atomic_add(grid_atomic2, 1).to(tl.int32)
            tl.atomic_add(grid_atomic, -nchunks * BLOCK_SIZE_SP)


def _fused_chunk_state_state_passing_fwd_concurrent(B, x, dt, dA_cumsum, out_dtype, initial_states=None, seq_idx=None, states=None, states_in_fp32=True):
    # original setup from chunk state
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype)

    # original setup from state passing
    _, _, _, dim1, dim2 = states.shape
    dim = dim1 * dim2
    assert dA_cumsum.shape[:3] == (batch, nheads, nchunks)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim1, dim2)
    assert chunk_size is not None
    if seq_idx is not None:
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, nheads, dim1, dim2), device=states.device, dtype=out_dtype)
    final_states = torch.empty((batch, nheads, dim1, dim2), device=states.device, dtype=torch.float32)


    # grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
    #                 batch * nchunks, nheads)
    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    # TODO: should probably be part of configs or something
    # step2_count = 8
    grid = lambda META: (META['SKIP_THREADBLOCK_COUNT'] + triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']) * batch * nchunks * nheads, )
    # grid = lambda META: (sm_count,)
    # actual_grid_size_raw = torch.cuda.get_device_properties("cuda").multi_processor_count
    # actual_grid_size = lambda META: (actual_grid_size_raw,) # TODO: in all this new code, I hardcoded device "cuda"
    grid_atomic = torch.zeros((1), device="cuda")
    grid_atomic2 = torch.zeros((1), device="cuda")
    chunk_work_done = torch.zeros((batch, nheads), dtype=torch.int32, device="cuda") # TODO: could probably be int8

    print("WARNING: this kernel can deadlock if SKIP_THREADBLOCK_COUNT is too high!!! If it takes too long, kill it forcefully")

    with torch.cuda.device(x.device.index):
        _fused_chunk_state_state_passing_fwd_kernel_concurrent[grid](#actual_grid_size](
            grid_atomic,
            grid_atomic2,
            chunk_work_done,
            # actual_grid_size_raw,
            sm_count, # min concurrent threadblocks, we know that at least this many threadblocks can run concurrently
            # step2_count,

            x, B, states, dt, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nheads // ngroups, nheads, nchunks,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4), states.stride(-1), # TODO: some strides are assumed decreasing
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),

            out, final_states, dA_cumsum, initial_states,
            out.stride(0), out.stride(1), out.stride(2), out.stride(4),
            final_states.stride(0), final_states.stride(1), final_states.stride(3),
            dA_cumsum.stride(-1) * (chunk_size - 1), # offset to index into the last element along last dim
            *((initial_states.stride(0), initial_states.stride(1), initial_states.stride(3))
              if initial_states is not None else (0, 0, 0)),
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    print("the kernel finished without deadlocking")
    return out, final_states
