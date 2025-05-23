# adapted from https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
 
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def sq_kernel(x_ptr,  # *Pointer* to first input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    # get threadblock id
    pid = tl.program_id(axis=0)
    

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * x
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def sq_kernel_persistent(threadblock_counter, # for scheduling "virtual/manual/fake" threadblocks on "real" threadblocks
               x_ptr,  # *Pointer* to first input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    # to detect that we are finished allocating work
    # we do this in the kernel instead of passing as a parameter,
    # because if you autotune, BLOCK_SIZE is only known at runtime
    target_grid_size = tl.cdiv(n_elements, BLOCK_SIZE) 

    should_exit = False # triton doesn't allow break or return, so this is just a workaround
    while not should_exit:
        # get threadblock id
        pid = tl.atomic_add(threadblock_counter, 1)
        if pid >= target_grid_size:
            should_exit = True # exit the entire kernel

        if not should_exit:
            # most of the original kernel code is here

            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)

            mask = offsets < n_elements
            
            x = tl.load(x_ptr + offsets, mask=mask)
            output = x * x
            tl.store(output_ptr + offsets, output, mask=mask)


def sq(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()

    # basic grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    sq_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

def sq_persistent(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()

    # persistent grid
    # we can run at least as many threadblocks as SMs
    SM_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = lambda meta: (SM_count, )

    # we need a single number of scratch space for counting how many "virtual/manual/fake" threadblocks have run
    threadblocks_run_atomic = torch.zeros((1,), device=DEVICE, dtype=torch.int32)
    
    sq_kernel_persistent[grid](threadblocks_run_atomic, x, output, n_elements, BLOCK_SIZE=1024)
    return output



if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    output_torch = x * x
    output_triton = sq(x)
    output_triton_persistent = sq_persistent(x)
    print(output_torch)
    print(output_triton)
    print(output_triton_persistent)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')
    print(f'The maximum difference between torch and triton persistent is '
        f'{torch.max(torch.abs(output_torch - output_triton_persistent))}')
