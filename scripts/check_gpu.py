"""Verifica que PyTorch ve la GPU y que bf16 funciona."""
import time

import torch

print("torch:", torch.__version__)
print("CUDA de torch:", torch.version.cuda)
print("cuda disponible:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA no está disponible")

props = torch.cuda.get_device_properties(0)
print("GPU:", props.name)
print(f"VRAM total: {props.total_memory / 1024**3:.1f} GiB")
print("bf16 soportado:", torch.cuda.is_bf16_supported())

a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.matmul(a, b)  # calentamiento
torch.cuda.synchronize()
t0 = time.perf_counter()
c = torch.matmul(a, b)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"matmul 4096x4096 bf16: {dt * 1000:.2f} ms, dtype={c.dtype}, "
      f"finito={bool(torch.isfinite(c.float()).all())}")
