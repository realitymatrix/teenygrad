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

//! An int8 `T::dot` (i8 x i8 -> i32) compiled for x86 and run on this host, against an
//! exact host reference. With AVX-VNNI (or AVX512-VNNI+VL) teenyc lowers it to
//! `vpdpbusd`; the result must still be exact, extremes of both operands included.
//!
//! `C[m, n] = sum(k) A[m, k] * W[n, k]`: the weights are row-major `[N, K]`, as a
//! linear layer stores them, loaded as `[BLOCK_N, BLOCK_K]` tiles and transposed for
//! the dot -- the shape the VNNI lowering folds without shuffles.

#![cfg(target_arch = "x86_64")]

use dotenv::dotenv;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;
use teeny_macros::kernel;
use teeny_triton::triton::*;
use teeny_x86::compiler::compile_kernel;
use teeny_x86::compiler::target::{Capability, Target};
use teeny_x86::device::context::X86DeviceInfo;
use teeny_x86::device::program::X86Program;
use teeny_x86::device::{X86Device, X86LaunchConfig};

/// Grid: `(M / BLOCK_M, N / BLOCK_N)`. Shapes must be multiples of the blocks.
#[kernel]
pub fn int8_matmul_nt_forward<T: Triton, const BLOCK_M: i32, const BLOCK_N: i32, const BLOCK_K: i32>(
    a_ptr: InPtr<T::Pointer<i8>>,
    w_ptr: InPtr<T::Pointer<i8>>,
    c_ptr: OutPtr<T::Pointer<i32>>,
    M: i32,
    N: i32,
    K: i32,
) {
    let pm = T::program_id(Axis::X);
    let pn = T::program_id(Axis::Y);
    let a_desc = T::make_tensor_descriptor(a_ptr, &[M, K], &[K, 1], &[BLOCK_M, BLOCK_K], None);
    let w_desc = T::make_tensor_descriptor(w_ptr, &[N, K], &[K, 1], &[BLOCK_N, BLOCK_K], None);
    let mut acc = T::zeros::<i32>(&[BLOCK_M, BLOCK_N]);
    for k in 0..K / BLOCK_K {
        let a = T::load_tensor_descriptor(a_desc, &[pm * BLOCK_M, k * BLOCK_K]);
        let w = T::load_tensor_descriptor(w_desc, &[pn * BLOCK_N, k * BLOCK_K]);
        let w_t = T::trans(w, &[1, 0]);
        acc = T::dot::<i8, i32>(a, w_t, Some(acc), InputPrecision::IEEE, None);
    }
    let c_desc = T::make_tensor_descriptor(c_ptr, &[M, N], &[N, 1], &[BLOCK_M, BLOCK_N], None);
    T::store_tensor_descriptor(c_desc, &[pm * BLOCK_M, pn * BLOCK_N], acc);
}

fn run(m: usize, n: usize, k: usize, block_m: i32) -> anyhow::Result<()> {
    const BLOCK_N: i32 = 32;
    const BLOCK_K: i32 = 64;
    let kernel = Int8MatmulNtForward::new(block_m, BLOCK_N, BLOCK_K);
    let path = compile_kernel(&kernel, &Target::new(Capability::Native), true, false)?;
    let asm = std::fs::read_to_string(path.replace(".so", ".s")).unwrap_or_default();
    eprintln!(
        "int8 dot {m}x{n}x{k}: vpdpbusd x{} in the kernel",
        asm.matches("vpdpbusd").count()
    );

    // Deterministic values covering the i8 extremes on both sides.
    let a: Vec<i8> = (0..m * k)
        .map(|i| match i % 7 {
            0 => -128,
            1 => 127,
            _ => ((i * 37 + 11) % 256) as u8 as i8,
        })
        .collect();
    let w: Vec<i8> = (0..n * k)
        .map(|i| match i % 5 {
            0 => 127,
            1 => -128,
            _ => ((i * 53 + 7) % 256) as u8 as i8,
        })
        .collect();

    let device = X86Device::new(X86DeviceInfo::default());
    let mut a_buf = device.buffer::<i8>(a.len())?;
    let mut w_buf = device.buffer::<i8>(w.len())?;
    let c_buf = device.buffer::<i32>(m * n)?;
    a_buf.to_device(&a)?;
    w_buf.to_device(&w)?;
    let program = X86Program::<Int8MatmulNtForward>::try_new(&path)?;
    device.launch(
        &program,
        &X86LaunchConfig::new([(m / block_m as usize) as u32, (n / BLOCK_N as usize) as u32, 1]),
        (
            a_buf.as_device_ptr(),
            w_buf.as_device_ptr(),
            c_buf.as_device_ptr(),
            m as i32,
            n as i32,
            k as i32,
        ),
    )?;
    let mut c = vec![0i32; m * n];
    c_buf.to_host(&mut c)?;

    for i in 0..m {
        for j in 0..n {
            let want: i32 = (0..k)
                .map(|kk| i32::from(a[i * k + kk]) * i32::from(w[j * k + kk]))
                .sum();
            assert_eq!(c[i * n + j], want, "C[{i}, {j}] for {m}x{n}x{k}");
        }
    }
    Ok(())
}

#[test]
fn int8_dot_one_row_is_exact() -> anyhow::Result<()> {
    dotenv().ok();
    run(1, 64, 256, 1)
}

#[test]
fn int8_dot_several_rows_is_exact() -> anyhow::Result<()> {
    dotenv().ok();
    run(16, 96, 2304, 16)
}
