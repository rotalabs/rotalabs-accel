# Modules

Drop-in `nn.Module` replacements with optimized kernels.

These modules provide the same interface as their PyTorch counterparts but use optimized Triton kernels when available.

## TritonRMSNorm

::: rotalabs_accel.kernels.normalization.TritonRMSNorm

## SwiGLU

::: rotalabs_accel.kernels.activations.SwiGLU

## RotaryEmbedding

::: rotalabs_accel.kernels.rope.RotaryEmbedding

## Int8Linear

::: rotalabs_accel.kernels.gemm.Int8Linear

## QuantizedLinear

::: rotalabs_accel.quantization.symmetric.QuantizedLinear
