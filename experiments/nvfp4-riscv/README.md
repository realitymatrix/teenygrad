# NVFP4 on RISC-V

Can a RISC-V CPU consume NVIDIA's NVFP4 weights? Yes, by software dequant, and
bit-exactly. This directory holds the kernel, a portable reference, and the
real test vectors that prove it.

Native NVFP4 *arithmetic* is a different question, and the answer there is no:
FP4 tensor ops are a Blackwell feature (tcgen05) and no CPU ISA has them.
Dequant-to-float is the only path. That is not the same as giving up the memory
win, as long as you dequantize per block into a tile at matmul time rather than
expanding the whole tensor to BF16 in RAM.

## The format

As shipped in `thinkingmachines/Inkling-Small-NVFP4`:

| tensor | dtype | meaning |
| --- | --- | --- |
| `*_weight` | U8 `[experts, rows, bytes]` | two E2M1 nibbles per byte, low nibble is the even element |
| `*_weight.scale` | F8_E4M3 `[experts, rows, bytes/8]` | one scale per **16** elements |
| `*_weight.scale2` | F32 `[experts]` | one global scale per expert |

`value = E2M1 x e4m3(scale) x scale2`

The 16-element block is what distinguishes NVFP4 from MXFP4, which uses 32
elements and an E8M0 scale. `hf_quant_config.json` holds the embeddings, norms,
attention and sconv layers out of quantization entirely.

E2M1 is `s:1 e:2 m:1`, decoding to 0, 0.5, 1, 1.5, 2, 3, 4, 6 and negatives.

## Files

    nvfp4.c        portable scalar reference (builds for any target)
    nvfp4_rvv.c    standalone RVV 1.0 dequant kernel
    nvfp4_gemm.c   dequant FUSED into a GEMM tile loop (the production shape)
    gemm_test.c    fused vs unfused: accuracy and amortisation
    harness.c      bit-exactness check, RVV vs scalar
    bench.c        throughput, scalar vs RVV
    ref.py         independent numpy reference
    hdr.py grab.py fetch one tensor slice from Hugging Face by byte range
    testdata/      19 KB of real weights: expert 0, rows 0-7 of layer 20 w13

## Running it

    make check     # numpy -> scalar riscv -> RVV at all VLENs, plus the fused GEMM
    make bench     # dequant throughput (meaningless under qemu; see below)
    make gemm      # fused vs unfused across tile widths

Needs `riscv64-linux-gnu-gcc` and `qemu-riscv64-static`. Verified with gcc 14.2.

## Why the kernel looks the way it does

**No lookup table, no `vrgather`.** E2M1 to binary32 is pure bit assembly: sign
to bit 31, `exp = e + 126`, mantissa bit `m << 22`, plus one masked merge for
`e==0, m==1`, which means 0.5. A 16-entry LUT would need `vrgather` over a
register group large enough to hold 16 FP32 lanes, which quietly depends on
VLEN. The bit version makes no element-count assumption, so it passes unchanged
from VLEN 128 to 1024.

**`vsseg2e32` for the interleave.** The low and high nibbles produce two
vectors whose elements must alternate in memory. A segment store does that in
one instruction instead of a scatter.

**The block scale is folded with `scale2` once per block** and fetched with
`vluxei32`.

## Fusing dequant into the matmul

Standalone dequant costs roughly **21 vector element-ops per weight**, against
**2 FLOPs** for the multiply-add that consumes it. At batch 1 nothing amortises
that: every weight is unpacked, used once, and discarded, so dequant is about
10x the arithmetic of the dot product itself.

`nvfp4_gemm.c` fuses the two. A weight is unpacked once per `k` and reused
across all `B` columns of the tile, and the dequantised row never reaches
memory. The cost per useful MAC therefore falls as `21/B + 2` rather than
staying at `23`.

Two implementation notes:

- RVV vector types are **sizeless**, so accumulators cannot be held in an
  array. The column loop is unrolled by macro and the tile width is a
  compile-time constant; `GEN(BN, REP)` instantiates widths 1, 2, 4 and 8.
  Verified with `objdump` that none of the four spills.
- Instead of interleaving the two nibble vectors back into `k` order with
  `vsseg2e32`, the fused kernel keeps them apart and uses a segment **load**
  (`vlseg2e32`) to split X the same way: even lanes meet low nibbles, odd meet
  high. Two accumulators per column, summed at the end. That removes the
  interleave from the inner loop entirely.

Accuracy is no longer bit-exact against the unfused path and cannot be: vector
lanes accumulate in a different order than a sequential scalar sum. Measured
relative error is 1.9e-06 at B=1 rising to 3.9e-05 at B=8, which is ordinary
FP32 reassociation over a 4096-term dot product.

## Two traps

1. **Fold `scale * scale2` per block in every implementation.** Computing
   `(E2M1 * scale) * scale2` in one place and `E2M1 * (scale * scale2)` in
   another gives 1-ULP mismatches from multiply association alone. This looks
   exactly like a vector bug and is not one: the giveaway is that the mismatch
   count is identical at every VLEN.
2. **`-0` is not `-0.0`.** Writing `-0` as a Python int in the E2M1 table
   yields `+0.0`, so nibble `0x8` silently mismatches. The C side was right.

## Status

Bit-exact: numpy, scalar RISC-V under qemu, and RVV at VLEN 128/256/512/1024
all agree on all 32768 test values.

Fused GEMM: correct at tile widths 1, 2, 4, 8 (see error figures above).

Throughput: **unmeasured.** Under qemu the RVV path reports 0.53-0.65x of
scalar, which says only that qemu emulates vector instructions in software. The
one real signal is that it scales with VLEN (60.9 -> 73.6 Melem/s from 128 to
512 bits). A meaningful number needs the SpacemiT K3 board; `nvfp4_bench` is
what to run there.

## Scale, for context

Inkling-Small is 276B total / 12B active, 170.8 GB in NVFP4. It does not fit
the K3. MoE sparsity cuts compute per token, not resident weights: routing is
per token and per layer, so all 276B must stay reachable. Running it on the
board would mean streaming experts off storage, which puts it in minutes per
token at best.

The test vectors here were pulled by HTTP range request against the safetensors
shard headers: **19 KB downloaded, not 170 GB**. `hdr.py` reads the JSON header
at the front of a shard for byte offsets; `grab.py` fetches one tensor slice.
That trick is reusable for any safetensors model.
