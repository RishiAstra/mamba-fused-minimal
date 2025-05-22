import torch

# TODO: check if these are the dims we want to test
def generate_dummy_data(
    n_groups = 1,
    dstate = 128,
    nheads = 80, 
    headdim = 64, 
    seq_len = 2048,
    dtype = torch.bfloat16,
    device = torch.device('cuda'),
    chunk_size = 256,
    batch_size = 1,
):
    
    torch.manual_seed(42)

    x = torch.randn((batch_size, seq_len, nheads, headdim), dtype=dtype, device=device)
    A = torch.rand((nheads,), dtype=dtype, device=device) * -1 # sensitive
    B = torch.randn((batch_size, seq_len, n_groups, dstate), dtype=dtype, device=device)
    C = torch.randn((batch_size, seq_len, n_groups, dstate), dtype=dtype, device=device)
    D = torch.randn((nheads,), dtype=dtype, device=device)
    dt = torch.rand((batch_size, seq_len, nheads), dtype=dtype, device=device) * 0.5 + 0.5 # sensitive
    dt_bias = torch.rand((nheads,), dtype=dtype, device=device) * 0.5 - 0.25 # sensitive
    
    return x, A, B, C, D, dt, dt_bias