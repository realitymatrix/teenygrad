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

/// Ahead-of-time kernel compilation.
pub mod aot;
/// x86-64 kernel compilation driver (`teenyc` / LLVM path).
pub mod driver;
/// Compiling a `teeny-core` graph's kernels.
pub mod graph;
/// AOT compile options (`--options` on the AOT compile CLI).
pub mod options;
/// `-C target-cpu` chip identifiers and the [`teeny_core::compiler::Target`] impl for x86-64.
pub mod target;

/// Compiles a kernel to a x86-64 ELF shared library via `teenyc` (see [`driver::compile_kernel`]).
pub use driver::compile_kernel;

/// `--target` triple `teenyc` uses for the x86-64 backend, set on `LlvmCompiler` via
/// [`teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler::with_target_triple`].
///
/// A dedicated triple (not one of the general-purpose `x86_64-unknown-linux-gnu` targets) that exists purely
/// to be compiled through the `mlir` codegen backend's x86-64 path -- see
/// `rustc_target::spec::targets::x86_64_teeny_kernel` in the `teeny` compiler fork.
pub const TARGET_TRIPLE: &str = "x86_64-teeny-kernel";
