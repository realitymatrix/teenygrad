# Inkling-2-Mini on x86: a native teenygrad backend, and the Intel NPU

Inkling-2-Mini ran only on the SpacemiT K3 ([inkling2-k3-performance.md](inkling2-k3-performance.md)).
This adds an x86-64 backend to teenyc and teenygrad, runs the model natively on an Intel
Core Ultra laptop, checks it against the transformers reference, and measures what the
laptop's NPU can and cannot take.

## Where it ended up

Greedy, `--effort none`, the K3 prompts. teeny-bench is `--model inkling2:mini -p 64 -n 32 -r 2`.

| | x86 (this work) | K3 |
|---|---|---|
| Load, checkpoint in page cache | **10.4-10.9 s** | 11-13 s |
| Load, checkpoint from disk | up to 99 s | |
| Prompt, chat | **14.4 tokens/s** | 5.7-6.0 |
| Prompt, teeny-bench pp64 | **16.2-16.4 tokens/s** | 5.5 |
| Decode, chat | **9.3-9.4 tokens/s** (8 threads) | 3.2-3.3 |
| Decode, teeny-bench tg32 | **9.2-10.1 tokens/s**, p50 98-105 ms | 3.2, p50 314 ms |
| Matches the reference | **32 of 32 tokens, both prompts** | |

Decode is about 3x the K3, prompts about 3x. The machine's ceiling for decode is about 28
tokens/s (below); 15-20 looks reachable.

## The machine

- **CPU:** Intel Core Ultra X9 388H (Panther Lake), 16 cores (P, E and low-power E).
  AVX2, FMA, F16C, GFNI, **AVX-VNNI**; no AVX-512, no AMX, no AVX-VNNI-INT8.
- **Memory:** 64 GB LPDDR5X-8533. Measured read bandwidth with AVX2 loads: 28 GB/s on one
  thread, 51 on two, 83 on four, **101 GB/s** on 16.
- **NPU:** Intel AI Boost, NPU 5 (architecture 5010), driver 32.0.100.5600, OpenVINO
  2026.4.1: 25 TFLOPS FP16, 50 TOPS INT8, no BF16/FP8/FP4.
- **OS:** Windows 11; teenygrad and teenyc run in WSL2 (Ubuntu 26.04, 54 GB given to it).
  The NPU is visible only from Windows: WSL2 has no `/dev/accel`.

## Correctness

The transformers reference (`scripts/inkling2-ref-model.py`, the real weights through
transformers' own modules) took hours per prompt when generating step by step, so the
check is teacher-forced: one forward pass over the prompt plus teenygrad's 32 generated
tokens, asserting that at every position the reference's argmax is the token teenygrad
generated. **32 of 32 match on both prompts.** The prompt token counts (25 and 23)
match teenygrad's, and the re-encoded replies decode back to exactly the printed text.

Every later change kept the replies byte for byte identical to the checked ones:

- capital of France: `The user asks for the capital of France, one word. Need answer
  only. Capital of France is Paris. Ensure one word. No extra.` (32 tokens of thinking);
- sky: `The user asks for two sentences explaining why the sky is blue. Need answer concise.
  Need ensure accuracy: sunlight contains all wavelengths; air`.

All 63 Inkling-2 kernel tests pass natively on x86 (in about 2-10 s, against minutes
under qemu).

## The steps

| Step | Decode | Prompt |
|---|---|---|
| first native run | 1.2 tokens/s | 2.2-2.5 |
| `tt.gather` to `vpshufb` | 6.4-7.0 (chat) | 11.7-13.7 (chat) |
| int8 dot to VNNI, first NVFP4 kernel | level | level |
| VNNI kernel: contiguous activations, one dot, broadcasts hoisted | p50 120-123 ms | 13.7 (pp64) |
| E4M3 scales in one shift | p50 118-119 ms | 14.8-15.0 (pp64) |
| FMA in x86 kernels | p50 -3.5% | +6% |
| spin pool without a mutex | p50 107-116 ms | 14.2-15.7 (pp64) |
| signed int8 linear weights | **p50 98-105 ms** | **16.2-16.4** (pp64) |

### 1. An x86 backend

**teenyc** (`feat/x86-cpu-backend`). teenyc already had an abstract TritonCPU backend
(`CpuBackend`) that picks x86 vector lowerings, AVX-512 and AMX paths from the feature
set; `RiscvBackend` is one subclass. `X86Backend` is another:
- it emits an ELF object for the requested LLVM cpu and links a shared library with
  `ld.lld -m elf_x86_64`;
- the cpu is `x86-64-v3` by default, or `native`, resolved to the host;
- math calls go to libm (no host SLEEF yet).

Around it:
- a new target, `x86_64-teeny-kernel`, routes to the mlir backend as `riscv64-generic`
  does, so ordinary x86 builds are untouched;
- `TargetBackend::X86 = 3` with its options on both sides of the FFI.

**teenygrad** (`feature/inkling2-x86-npu`):
- **`drivers/teeny-x86`**: teeny-riscv's host code (buffers, programs, spin pool, kernel
  cache) with the RISC-V `lp64d` launcher replaced by a System V one. That means six
  integer registers, eight SSE registers without NaN-boxing, and floats past `xmm7` going
  to the stack, not to integer registers. A unit test calls a real `extern "C"` function
  through the fixed signature with arguments spilling to the stack. There is no qemu and
  no A100 cluster.
- **An `x86` feature** from teeny-runtime through teeny-test, teeny-kernels, teeny-llm,
  teeny-api and teeny-bench, so the model builds with
  `--no-default-features --features inkling2,x86`.
- **teeny-compiler** names x86 kernels `.so`, as it does RISC-V ones.

The first run decoded at 1.2 tokens/s.

### 2. `tt.gather` to `vpshufb`

`perf` put 45% of the time in the NVFP4 expert kernel. Its 16-entry E2M1 table lookup
(`tt.gather`) had no x86 lowering: it went through a stack buffer and a `vector.gather`,
which LLVM turned into a lane-by-lane loop of address extracts and byte inserts.

teenyc now lowers it in registers:
- a table of at most 16 bytes becomes `vpshufb`, 32 lanes per ymm with the table
  repeated into both halves;
- a table of at most 8 dwords becomes `vpermd`/`vpermps`.

Decode went from 1.2 to 6.4-7.0 tokens/s, prompts from 2.2 to 11.7-13.7. Output was
unchanged.

### 3. int8 dot to VNNI

Both remaining hot kernels (NVFP4 experts and the int8 linears) were compute-bound, not
memory-bound. A new teenyc pass, `ConvertDotToVNNI`, lowers a rank-2 `tt.dot` with
i8 x i8 -> i32 operands (K % 4 == 0, N % 8 == 0) to `vpdpbusd`:
- The columns are the lanes. Four k of each column are packed into a 32-bit lane, and
  four k of each row of A are broadcast. One instruction covers 8 columns x 4 k.
- `vpdpbusd`'s first operand is unsigned, so B goes in as B XOR 0x80, and
  128 x sum(A[m, :]) is subtracted once per row. The result is exact.
- A transpose of a row-major `[N, K]` tile, or a reshape of a `[0, 2, 1]` transpose of
  `[K / 4, N, 4]`, folds into a bitcast. Anything else gets a byte interleave.
- It runs on x86 targets with AVX-VNNI, or AVX512-VNNI with VL. LLVM 23's
  `vpdpbusd.256` takes `<32 x i8>` operands.

`drivers/teeny-x86/tests/test_int8_dot_x86.rs` checks it exactly against a host
reference, i8 extremes included. It covers 1x64x256 and 16x96x2304 with transposed
weights, an untransposed case, and the NVFP4 kernel's nibble decode on its own.

**`expert_gemv_nvfp4_vnni_forward`** computes the same exact integer projection as
`expert_gemv_nvfp4_forward`, with each 16-element block as one K = 16 int8 dot.
- **Weight layout.** Per row block and 16-element block, the weights are
  `[2][rows][4]` bytes. The low nibbles hold elements {0-3, 8-11} of each row, the high
  nibbles {4-7, 12-15}. So `byte & 15` and `byte >> 4` are each two packed chunks with no
  interleaving; the activations are loaded contiguously and reordered to match in
  registers.
- **Loading.** `vnni_interleave` derives this layout from the format-2 bytes at load (slot
  `weight_vnni`). **The converted checkpoint is unchanged** and the K3 build reads exactly
  what it did.
- **Default.** It is the x86 default; `INKLING2_VNNI_EXPERTS=0` restores the other
  kernel. Both pass all 63 kernel tests.

The first version was level with the old kernel. A cycle breakdown showed why: 33% stack
spills, 30% scalar code from the activations being picked element by element, 18%
broadcasts repeated per 8-lane group. The dot and the table lookups were about 5%.
Fixing those (contiguous activations, one dot per block instead of two, broadcasts once
per chunk) took prompts from about 12 to 13.7 tokens/s and decode p50 from about 128
to about 121 ms.

### 4. E4M3 block scales in one shift

With those gone, decoding the 32 E4M3 block scales of each block was 17% of the kernel.
An E4M3 byte's low seven bits shifted to bit 20 form an f32 exactly 2^-120 times its
value, subnormals included: the exponent and mantissa land in the right fields and the
biases differ by 120. So the block scale is `bitcast((b & 0x7F) << 20) * (2^120 * es)`,
with the factor folded into the expert scale once. Powers of two multiply exactly, so
every scale is bit for bit what the old compare, convert, multiply and select gave, in
four instructions per eight scales instead of about nine. Prompts went to 14.8-15.0
tokens/s and decode p50 to 118-119 ms.

### 5. Launch overhead, FMA and signed int8 weights

A fresh budget put 21 ms of a decode token outside the kernels, about 8 us on each of
about 2,470 launches. Each change below was measured interleaved against the build
before it (three rounds of teeny-bench), and each left both replies byte for byte
identical to the reference-checked ones.

- **The spin pool handed each job over through a mutex.**
  - Every worker locked it at the same instant on every launch; `perf` showed it as
    `Mutex::lock_contended`.
  - The job is now the two words of its fat pointer in atomics, ordered by the epoch's
    release/acquire pair.
  - Decode p50 went from 127-131 to 107-116 ms (-14%).
  - teeny-riscv's pool has the same mutex.
- **FMA.**
  - `X86Backend` creates its TargetMachine with `AllowFPOpFusion = Fast`, so a
    multiply feeding an add becomes one `vfmadd`. It rounds once instead of twice.
  - Prompts +6%, decode p50 -3.5%.
  - `TEENYC_X86_NO_FMA=1` turns it off.
- **Signed int8 linear weights.**
  - The int8 linears stored each weight as `q + 128` and subtracted 128 after widening,
    a quarter of the inner loop.
  - On x86 the loader flips the bytes to `q` (XOR 0x80) and
    `linear_lanes_i8_signed_forward` sign-extends instead. The arithmetic is identical.
  - Prompts +19%, decode p50 about -7%, load time unchanged.
  - `INKLING2_SIGNED_I8=0` turns it off.
- **Tried, no clear gain:** thread count (6-10) and spin time (50-1000 us). Short spins
  are worse; the rest is within noise.
- **The laptop's "AI Engine" power profile is 25-35% slower than the previous one**
  (decode p50 146-162 ms against 111-123 ms) and gets slower as it runs. It favours the
  NPU; teenygrad runs on the CPU. Use the Performance or Balanced profile.

## Threads

Panther Lake has P, E and low-power E cores, and WSL cannot pin threads to physical
cores. Every launch waits for its slowest thread, so:
- **decode** is best at **8 threads** (`TEENY_X86_THREADS=8`);
- **prompts** at 8-14;
- **16 threads is slower even on an idle machine.** tg32 drops to 2.6 tokens/s.

Runs vary by about +/-10% in decode, with longer p90/p99 tails than the K3.

## Where a decode token goes

Per decode token (difference of a 1-token and a 101-token run, `TEENY_RUNTIME_PROFILE`,
8 threads, before the VNNI kernels). The experts have since dropped by roughly the
difference in p50, about 10 ms.

| | ms | |
|---|---|---|
| experts (NVFP4 + MXFP8) | 48 | about 38 GB/s |
| attention projections (int8) | ~35 | about 37 GB/s |
| unembedding (int8) | 8.7 | about 53 GB/s |
| ~2,300 small ops | ~30 | ~25 us each |
| gaps between launches | ~22 | spin pool |

A token reads about 3.6 GB of weights (layers 6-17 run three times). At 101 GB/s that
is a **floor of about 36 ms, or about 28 tokens/s.** The kernels are at 37-53 GB/s of
that. 15-20 tokens/s looks reachable with:
- the experts and int8 linears streaming at 70% of bandwidth;
- the small ops fused, as the K3 doc's SGLang/vLLM list describes.

## The NPU

Measured with OpenVINO 2026.4.1 on Windows (the NPU is not reachable from WSL2).

| | NPU | CPU (OpenVINO) |
|---|---|---|
| Dense int8 weights, one 1.07 GB graph | 11.3 ms, **95 GB/s** | 13.4 ms, 80 GB/s |
| Smallest call, 17 MB | 0.58 ms (fixed cost ~0.4 ms) | 0.33 ms |
| A layer's qkv + o, 1 token | 0.87 ms | **0.46 ms** |
| A layer's qkv + o, 16 tokens | **0.99 ms** | 2.14 ms |
| Unembedding, 16 tokens | **11.4 ms** | 39.0 ms |
| 8 of 128 experts gathered in-graph, one layer | 6.8-8.4 ms | 12-13 ms |

- **Experts cannot go to the NPU.** Selecting 8 of 128 experts inside an NPU graph costs
  6-8 ms a layer. A token runs 48 such layers, so that alone is about 300 ms. The NPU has
  no NVFP4 or FP8 either, so it would also mean re-quantising the experts, which changes
  the model's output.
- **Decode should stay on the CPU.** With about 100 dense calls a token, each paying the
  NPU's fixed ~0.4 ms, offloading the projections costs about 40 ms where the CPU spends
  about 22.
- **Prompts can gain.** At 16 tokens a call the NPU is 2-3.5x the CPU. But in a prompt
  chunk the experts are 63% of the time and the dense layers the NPU can take are 21%:
  attention projections 16%, controllers, router gate and unembedding 5%. So the gain is
  capped at about +20% on prompts.
  - It needs the NPU side to run on Windows: either a Windows build of teenygrad and its
    kernels, or a bridge from WSL. **Not built.**

## What did not work, or is left

- **The VNNI expert kernel's first version was not faster.** It ran level with the old
  kernel until the code-generation overheads above were fixed. My first measurements of
  it were wrong: the chat binary had not been rebuilt.
- **teenyc mis-shapes `tt.cat` of two 3-D tensors.** A cat of two `[2, 32, 4]` tensors
  comes out as `[4, 8, 64]`. The kernel concatenates flat instead; worth fixing in teenyc.
- **`T::dot` without an accumulator accumulates in its input type.** An i8 dot with
  `None` sums in i8. The kernel passes an explicit i32 zero.
- **The VNNI expert kernel still spills.** There are 94 stack references in its
  assembly; a smaller live set should help further.
- **The int8 attention linears** (about 35 ms of a decode token) convert every weight to
  f32 before multiplying. VNNI would need int8 activations, which changes the arithmetic,
  so it needs a quality check first.
- **The MXFP8 experts** (layers 20-23) have no x86-specific work yet.
- **Pre-existing, not from this work:**
  - 7 of 8 tests in `test_inkling2_model.rs` fail on host-side graph construction, before
    any kernel runs, on any backend. They feed f32 weights under names the graph now gives
    bf16 slots (`*.weight_bf16`), and their 24-wide stand-in config is not a multiple of
    the 16-element blocks the integer NVFP4 experts need.
  - `teeny-riscv` and `teeny-x86` do not build on their own: `teeny-core`'s
    `model/mod.rs:162` uses `std::path` without the `std` feature. A model build turns it
    on; the drivers' tests need `--features teeny-core/std`.
- **No x86 SLEEF.** Vector math calls libm, though no hot kernel calls it.

## Building and running

The compiler (`realitymatrix/teeny`, branch `feat/x86-cpu-backend`, from
`feat/rvv-tt-gather`) needs, beyond its README:
- **A local `bootstrap.toml`** with:
  - `llvm.download-ci-llvm = false`, because CI LLVM has no MLIR or LLD;
  - `rust.codegen-backends = ["llvm", "mlir"]`, which is what makes bootstrap add MLIR to
    the LLVM build;
  - `rust.lld = true`;
  - `rust.deny-warnings = false`, for an unused variable in `rustc_llvm/build.rs`.
- **Initialising the fork's own submodules by hand** (melior, mlir-sys, tblgen-rs, triton,
  sleef, llvm-project). Bootstrap does not, and their `git@` URLs need
  `url.https://github.com/.insteadOf`.
- **clang-22**, which `rustc_llvm/build.rs` hard-codes for Triton's NVIDIA backend.
- **nanobind** on `CMAKE_PREFIX_PATH`, for Triton's Python module (`triton.toml`).

Then:

```bash
./x.py build --stage 1 -j 4 --warnings warn
```

That took about 1.5 hours here.

```bash
export TEENYC_PATH=<teeny>/build/host/stage1/bin/teenyc TEENYC_CAPABILITY=native
cargo build --release -p teeny-llm --no-default-features --features inkling2,x86 \
  --example inkling2 --example inkling2_convert
cargo build --release -p teeny-bench --no-default-features --features inkling2,x86

inkling2_convert --model <hf> --out <conv>       # 221 s here (377 s on the K3)
inkling2_convert --model <conv> --add-int8       # 11 s
inkling2_convert --model <hf> --add-compact <conv>   # 13 s

TEENY_X86_THREADS=8 target/release/examples/inkling2 --model <conv> --effort none \
  --max-tokens 32 --prompt "What is the capital of France? Answer in one word."
TEENY_X86_THREADS=8 target/release/teeny-bench --model inkling2:mini --model-dir <conv> \
  -p 64 -n 32 -r 2
```

Tests:

```bash
cargo test --release -p teeny-llm --no-default-features --features inkling2,x86 \
  --test test_inkling2_kernels
cargo test --release -p teeny-x86 --features teeny-core/std
```

## Commits

teeny-rustc, `feat/x86-cpu-backend`:
- `2bf29f0f1e0` x86-64 CPU backend for TritonCPU kernels
- `d0279d5c31b` `tt.gather` to `pshufb`/`vpermd` on x86
- `baf2a9f6c2b` int8 dot to x86 VNNI (`vpdpbusd`)
- `309fbc5796b` fold reshape(transpose) in the VNNI dot
- `1abf6d4794e` broadcast each activation chunk once
- `a62ee9f9c13` FMA in x86 kernels

teenygrad, `feature/inkling2-x86-npu` (from `rebase/inkling2-on-functiongemma`):
- `580b29f1d` teeny-x86, the in-process x86-64 backend
- `6bf9204ee` x86 kernels named `.so`; Inkling-2 tests on x86
- `c44d82ba5` an exact int8 `T::dot` test on x86
- `fac3c862e` the NVFP4 expert kernel on int8 `T::dot`
- `07a3425cf` VNNI experts by default on x86
- `8eb33b5c5` E4M3 block scales in one shift
- `e0ae828a1` this document
- `9ba26282e` spin pool without a mutex
- `1df3be802` signed int8 linear weights on x86
- the update of this document for them
