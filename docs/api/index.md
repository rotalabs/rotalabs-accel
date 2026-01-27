# API Reference

This section contains the complete API reference for rotalabs-accel, auto-generated from the source code docstrings.

## Package Structure

```
rotalabs_accel/
├── kernels/
│   ├── normalization    # RMSNorm kernels
│   ├── activations      # SwiGLU kernels
│   ├── rope             # Rotary Position Embeddings
│   └── gemm             # INT8 GEMM kernels
├── quantization/
│   └── symmetric        # INT8 symmetric quantization
├── modules/             # Drop-in nn.Module replacements
└── utils/
    └── device           # Device detection utilities
```

## Quick Links

- [Normalization](kernels/normalization.md) - RMSNorm with optional residual fusion
- [Activations](kernels/activations.md) - SwiGLU activation
- [RoPE](kernels/rope.md) - Rotary Position Embeddings
- [GEMM](kernels/gemm.md) - INT8 W8A16 matrix multiplication
- [Quantization](quantization.md) - INT8 symmetric quantization utilities
- [Modules](modules.md) - Drop-in nn.Module replacements
- [Utilities](utils.md) - Device detection and helpers
