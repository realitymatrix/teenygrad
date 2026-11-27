# teeny-x86

The x86-64 CPU device backend for [teenygrad](https://teenygrad.org): Target/Capability types for
the `mlir`/Triton compiler backend's `x86_64-teeny-kernel` path, and a `libloading`-based runtime
that runs compiled kernels in-process on every core.

Kernels compile through the `teeny` compiler fork's `X86Backend`: their Triton bodies are lowered
through TritonCPU to LLVM IR, compiled for the requested x86-64 level (`x86-64-v3` by default,
optionally with AVX-VNNI, `x86-64-v4` or `native`) and linked into an ELF shared library with
`ld.lld`. `X86Device` dlopens it and calls the entry point once per program id, spread over a
spinning thread pool (`TEENY_X86_THREADS`, `TEENY_X86_SPIN_US`).

Structurally this is `teeny-riscv` with the RISC-V `lp64d` calling convention replaced by SysV
x86-64 (`native.rs`), and without qemu or the SpacemiT A100 cluster.
