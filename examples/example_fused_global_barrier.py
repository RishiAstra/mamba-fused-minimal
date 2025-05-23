# adapted from https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
 
import torch

import triton
import triton.language as tl

from example_persistent import add_kernel, add_kernel_persistent
from example_persistent2 import sq_kernel, sq_kernel_persistent

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# note that for these 2 add and sq example kernels,
# to just do both operations together before writing back the results
# however, this kernel fusion technique works with only the assumption
# that the first kernel must be fully finished before the second can run
@triton.jit
def add_sq_fused_kernel(threadblock_counter1, # for scheduling "virtual/manual/fake" threadblocks on "real" threadblocks
               threadblock_counter2,
               global_barrier_atomic,
               x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               intermed_ptr, # for the intermediate result between the 2 parts
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    ########################################
    # First Step (the first original kernel)
    ########################################
    # to detect that we are finished allocating work
    # we do this in the kernel instead of passing as a parameter,
    # because if you autotune, BLOCK_SIZE is only known at runtime
    target_grid_size = tl.cdiv(n_elements, BLOCK_SIZE) 

    should_exit = False # triton doesn't allow break or return, so this is just a workaround
    while not should_exit:
        # get threadblock id
        pid = tl.atomic_add(threadblock_counter1, 1)
        if pid >= target_grid_size:
            should_exit = True # exit the entire kernel

        if not should_exit:
            # most of the original kernel code is here

            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)

            mask = offsets < n_elements
            
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            intermed = x + y
            tl.store(intermed_ptr + offsets, intermed, mask=mask)

    ########################################
    # Global Barrier
    ########################################
    # increment counter
    at_bar_count = tl.atomic_add(global_barrier_atomic, 1) + 1
    num_needed = tl.num_programs(0) # get persistent grid size

    # busy-wait for barrier
    while at_bar_count < num_needed:
        at_bar_count = tl.atomic_add(global_barrier_atomic, 0) # atomic load


    ########################################
    # Second Step (the second original kernel)
    ########################################
    # to detect that we are finished allocating work
    # we do this in the kernel instead of passing as a parameter,
    # because if you autotune, BLOCK_SIZE is only known at runtime
    target_grid_size = tl.cdiv(n_elements, BLOCK_SIZE) 

    should_exit = False # triton doesn't allow break or return, so this is just a workaround
    while not should_exit:
        # get threadblock id
        pid = tl.atomic_add(threadblock_counter2, 1)
        if pid >= target_grid_size:
            should_exit = True # exit the entire kernel

        if not should_exit:
            # most of the original kernel code is here

            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)

            mask = offsets < n_elements
            
            intermed = tl.load(intermed_ptr + offsets, mask=mask)
            output = intermed * intermed
            tl.store(output_ptr + offsets, output, mask=mask)


def add_sq_unfused(x: torch.Tensor, y: torch.Tensor):
    intermed = torch.empty_like(x)
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()

    # basic grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    add_kernel[grid](x, y, intermed, n_elements, BLOCK_SIZE=1024)
    sq_kernel[grid](intermed, output, n_elements, BLOCK_SIZE=1024)
    return output

def add_sq_persistent_unfused(x: torch.Tensor, y: torch.Tensor):
    intermed = torch.empty_like(x)
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()

    # persistent grid
    # we can run at least as many threadblocks as SMs
    SM_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = lambda meta: (SM_count, )

    # we need a single number of scratch space for counting how many "virtual/manual/fake" threadblocks have run
    threadblocks_run_atomic1 = torch.zeros((1,), device=DEVICE, dtype=torch.int32)
    threadblocks_run_atomic2 = torch.zeros((1,), device=DEVICE, dtype=torch.int32)
    
    add_kernel_persistent[grid](threadblocks_run_atomic1, x, y, intermed, n_elements, BLOCK_SIZE=1024)
    sq_kernel_persistent[grid](threadblocks_run_atomic2, intermed, output, n_elements, BLOCK_SIZE=1024)
    return output

def add_sq_persistent_fused(x: torch.Tensor, y: torch.Tensor):
    intermed = torch.empty_like(x)
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()

    # persistent grid
    # we can run at least as many threadblocks as SMs
    SM_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = lambda meta: (SM_count, )

    # Note that all 3 atomic counters could probably be simplified into just 1 atomic counter
    # we need a single number of scratch space for counting how many "virtual/manual/fake" threadblocks have run
    threadblocks_run_atomic1 = torch.zeros((1,), device=DEVICE, dtype=torch.int32)
    threadblocks_run_atomic2 = torch.zeros((1,), device=DEVICE, dtype=torch.int32)
    # we also need a global barrier
    global_barrier_atomic = torch.zeros((1,), device=DEVICE, dtype=torch.int32)

    
    add_sq_fused_kernel[grid](threadblocks_run_atomic1, threadblocks_run_atomic2, global_barrier_atomic, x, y, intermed, output, n_elements, BLOCK_SIZE=1024)
    return output



if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    temp = x + y
    output_torch = temp * temp
    output_triton = add_sq_unfused(x, y)
    output_triton_persistent = add_sq_persistent_unfused(x, y)
    output_triton_fused = add_sq_persistent_fused(x, y)
    print(output_torch)
    print(output_triton)
    print(output_triton_persistent)
    print(output_triton_fused)
    print(f'The maximum difference between torch and triton unfused original is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')
    print(f'The maximum difference between torch and triton unfused persistent is '
        f'{torch.max(torch.abs(output_torch - output_triton_persistent))}')
    print(f'The maximum difference between torch and triton fused persistent is '
        f'{torch.max(torch.abs(output_torch - output_triton_fused))}')
