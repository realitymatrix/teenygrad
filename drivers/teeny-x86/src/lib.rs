/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! The x86-64 CPU device backend for [teenygrad](https://teenygrad.org) -- `Target`/`Capability`
//! types for the `mlir`/Triton compiler backend's `x86_64-teeny-kernel` path ([`compiler`]),
//! `teeny_core::device` implementations ([`device`]), and a `libloading` wrapper for calling a
//! kernel's exported symbol ([`runtime`]).
//!
//! Kernels are TritonCPU-lowered ELF shared libraries that [`device::X86Device`] dlopens and runs
//! in-process on an `x86_64` Linux host, spread over every core. Structurally a copy of
//! `teeny-riscv` with the RISC-V calling convention swapped for SysV x86-64, and without qemu or
//! the SpacemiT A100 cluster.

#![warn(missing_docs)]

/// `Target`/`Capability` types for compiling kernels via `LlvmCompiler`.
pub mod compiler;
/// `teeny_core::device` trait implementations -- the "device" is the local machine.
pub mod device;
mod elf;
/// Error types.
pub mod errors;
mod native;
// Used by the native launcher, which exists only on x86_64; built everywhere so its tests run.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
mod spin_pool;
/// Loading and calling a compiled kernel's shared library.
pub mod runtime;
