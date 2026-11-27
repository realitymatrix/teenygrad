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
use teeny_core::dtype::{AddOffsets, Tensor};
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

/// The same product with the weights stored `[K, N]` and loaded as `[BLOCK_K, BLOCK_N]`
/// tiles, no transpose: the VNNI lowering's interleaving path.
#[kernel]
pub fn int8_matmul_nn_forward<T: Triton, const BLOCK_M: i32, const BLOCK_N: i32, const BLOCK_K: i32>(
    a_ptr: InPtr<T::Pointer<i8>>,
    b_ptr: InPtr<T::Pointer<i8>>,
    c_ptr: OutPtr<T::Pointer<i32>>,
    M: i32,
    N: i32,
    K: i32,
) {
    let pm = T::program_id(Axis::X);
    let pn = T::program_id(Axis::Y);
    let a_desc = T::make_tensor_descriptor(a_ptr, &[M, K], &[K, 1], &[BLOCK_M, BLOCK_K], None);
    let b_desc = T::make_tensor_descriptor(b_ptr, &[K, N], &[N, 1], &[BLOCK_K, BLOCK_N], None);
    let mut acc = T::zeros::<i32>(&[BLOCK_M, BLOCK_N]);
    for k in 0..K / BLOCK_K {
        let a = T::load_tensor_descriptor(a_desc, &[pm * BLOCK_M, k * BLOCK_K]);
        let b = T::load_tensor_descriptor(b_desc, &[k * BLOCK_K, pn * BLOCK_N]);
        acc = T::dot::<i8, i32>(a, b, Some(acc), InputPrecision::IEEE, None);
    }
    let c_desc = T::make_tensor_descriptor(c_ptr, &[M, N], &[N, 1], &[BLOCK_M, BLOCK_N], None);
    T::store_tensor_descriptor(c_desc, &[pm * BLOCK_M, pn * BLOCK_N], acc);
}

#[test]
fn int8_dot_untransposed_weights_is_exact() -> anyhow::Result<()> {
    dotenv().ok();
    let (m, n, k) = (2usize, 64usize, 128usize);
    let kernel = Int8MatmulNnForward::new(2, 32, 16);
    let path = compile_kernel(&kernel, &Target::new(Capability::Native), true, false)?;
    let a: Vec<i8> = (0..m * k).map(|i| ((i * 37 + 11) % 256) as u8 as i8).collect();
    // b[k, n] = W[n, k], so the reference below is the same formula.
    let w: Vec<i8> = (0..n * k).map(|i| ((i * 53 + 7) % 256) as u8 as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| w[(i % n) * k + i / n]).collect();
    let device = X86Device::new(X86DeviceInfo::default());
    let mut a_buf = device.buffer::<i8>(a.len())?;
    let mut b_buf = device.buffer::<i8>(b.len())?;
    let c_buf = device.buffer::<i32>(m * n)?;
    a_buf.to_device(&a)?;
    b_buf.to_device(&b)?;
    let program = X86Program::<Int8MatmulNnForward>::try_new(&path)?;
    device.launch(
        &program,
        &X86LaunchConfig::new([1, (n / 32) as u32, 1]),
        (a_buf.as_device_ptr(), b_buf.as_device_ptr(), c_buf.as_device_ptr(), m as i32, n as i32, k as i32),
    )?;
    let mut c = vec![0i32; m * n];
    c_buf.to_host(&mut c)?;
    for i in 0..m {
        for j in 0..n {
            let want: i32 = (0..k).map(|kk| i32::from(a[i * k + kk]) * i32::from(w[j * k + kk])).sum();
            assert_eq!(c[i * n + j], want, "C[{i}, {j}]");
        }
    }
    Ok(())
}

/// The NVFP4 VNNI kernel's byte handling alone: a `[4][32][2]` byte block split into
/// nibbles, joined, looked up in a 16-entry table, transposed to `[16, 32]` and stored.
#[kernel]
pub fn nibble_block_forward<T: Triton>(w_ptr: InPtr<T::Pointer<u8>>, out_ptr: OutPtr<T::Pointer<i8>>)
where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::Pointer<u8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<u8>>>,
    T::Pointer<i8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i8>>>,
    T::Tensor<u8>: core::ops::BitAnd<u8, Output = T::Tensor<u8>>,
    T::Tensor<u8>: core::ops::Shr<u8, Output = T::Tensor<u8>>,
{
    // Table: code c -> c - 8, so the output shows exactly which nibble landed where.
    let lut = T::cast::<i32, i8>(T::arange(0, 16) - T::full(&[16], 8), None, false);
    let wb = T::load(w_ptr.add_offsets(T::arange(0, 256) + 0), None, None, &[], None, None, None, false);
    let codes = T::reshape(T::join(wb & 15u8, wb >> 4u8), &[512], false);
    let w = T::gather(lut, T::cast::<u8, i32>(codes, None, false), 0);
    let w = T::reshape(w, &[4, 32, 4], false);
    let w = T::reshape(T::trans(w, &[0, 2, 1]), &[512], false);
    T::store(out_ptr.add_offsets(T::arange(0, 512) + 0), w, None, &[], None, None);
}

#[test]
fn nibble_block_decodes_to_k_by_rows() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = NibbleBlockForward::new();
    let path = compile_kernel(&kernel, &Target::new(Capability::Native), true, false)?;
    let bytes: Vec<u8> = (0..256).map(|i| ((i * 29 + 3) % 256) as u8).collect();
    let device = X86Device::new(X86DeviceInfo::default());
    let mut w_buf = device.buffer::<u8>(256)?;
    let o_buf = device.buffer::<i8>(512)?;
    w_buf.to_device(&bytes)?;
    let program = X86Program::<NibbleBlockForward>::try_new(&path)?;
    device.launch(&program, &X86LaunchConfig::new([1, 1, 1]), (w_buf.as_device_ptr(), o_buf.as_device_ptr()))?;
    let mut out = vec![0i8; 512];
    o_buf.to_host(&mut out)?;
    // out[k][r]: element k = 4p + 2b + h of row r is nibble h of byte (p, r, b).
    let mut bad = Vec::new();
    for k in 0..16 {
        for r in 0..32 {
            let (p, b, h) = (k / 4, (k % 4) / 2, k % 2);
            let byte = bytes[(p * 32 + r) * 2 + b];
            let want = (if h == 0 { byte & 15 } else { byte >> 4 }) as i8 - 8;
            if out[k * 32 + r] != want && bad.len() < 8 {
                bad.push((k, r, out[k * 32 + r], want));
            }
        }
    }
    assert!(bad.is_empty(), "(k, row, got, want): {bad:?}");
    Ok(())
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
