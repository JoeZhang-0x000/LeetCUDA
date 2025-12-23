import torch
import triton
import triton.testing

import lib.sgemv as lib

torch.set_grad_enabled(False)
dtype = torch.float32
device = 'cuda'

SHAPES = [
    (128, 128),
    (1024, 128),
    (1024, 256),
    (4096, 128),
    (2048, 256),
    (128, 1024),
    (256, 1024),
    (1024, 1024),
]
SHAPE_LABELS = [f"{M}_{K}" for M, K in SHAPES]

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['M_K'],
        x_vals=SHAPE_LABELS,
        x_log=False,
        line_arg='provider',
        line_vals=['pytorch', 'custom_k32', 'custom_k128x4', 'custom_dispatch', ],
        line_names=['PyTorch_Matmul', 'Custom_k32_f32', 'Custom_k128_f32x4', 'Custom_Dispatch', ],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('black', '--')],
        ylabel='ms (Median)',
        plot_name='SGEMV Performance (ms)',
        args={"N": 1},
    )
])
def benchmark(M_K, N, provider):
    M, K = map(int, M_K.split("_"))
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)
    
    c_ref = torch.empty((M, N), dtype=dtype, device=device)
    torch.matmul(a, b, out=c_ref)

    if provider == 'custom_k32':
        c_out = torch.empty_like(c_ref)
        ms = triton.testing.do_bench(lambda: lib.sgemv_k32_f32(a, b, c_out))
        assert torch.allclose(c_out, c_ref, atol=1e-4, rtol=1e-4)

    elif provider == 'custom_k128x4':
        c_out = torch.empty_like(c_ref)
        ms = triton.testing.do_bench(lambda: lib.sgemv_k128_f32x4(a, b, c_out))
        assert torch.allclose(c_out, c_ref, atol=1e-4, rtol=1e-4)

    elif provider == 'custom_dispatch':
        c_out = torch.empty_like(c_ref)
        ms = triton.testing.do_bench(lambda: lib.sgemv(a, b, c_out))
        assert torch.allclose(c_out, c_ref, atol=1e-4, rtol=1e-4)

    elif provider == 'pytorch':
        c_out = torch.empty_like(c_ref)
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b, out=c_out))
    
    return ms

if __name__ == "__main__":
    benchmark.run(show_plots=False, print_data=True, save_path=None)