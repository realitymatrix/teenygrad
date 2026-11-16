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


//! `Trilu`: keeps the upper or lower triangle of the trailing two axes, zeroing the rest.
//!
//! Element `(r, c)` of each `rows x cols` matrix is kept when `c - r >= k` (upper) or
//! `c - r <= k` (lower). Both are `sign * (c - r) - sign * k >= 0` with `sign` +1 or -1,
//! so one kernel serves both with the sign and `-sign * k` passed as scalars.

use core::marker::PhantomData;

use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::Num;
use teeny_core::model::{RawPtr, RuntimeOp};
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// Forward: `y = x` where the element is in the kept triangle, else 0.
#[kernel]
pub fn trilu_forward<T, D: Num, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    n_elements: i32,
    rows: i32,
    cols: i32,
    sign: i32,
    neg_sign_k: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let p = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let in_bounds = p.lt(n_elements);

    // No integer remainder in the DSL: a coordinate is `q - (q / n) * n`.
    let q = p / cols;
    let c = p - q * cols;
    let q2 = q / rows;
    let r = q - q2 * rows;
    let keep = ((c - r) * sign + neg_sign_k).ge(0);

    let x = T::load(
        x_ptr.add_offsets(p),
        Some(in_bounds),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let y = T::where_(keep, x, T::zeros_like(x));
    T::store(y_ptr.add_offsets(p), y, Some(in_bounds), &[], None, None);
}

/// Runtime dispatch for [`trilu_forward`].
pub struct TriluRuntimeOp<D: Num + Send + Sync + 'static> {
    block_size: i32,
    rows: i32,
    cols: i32,
    sign: i32,
    neg_sign_k: i32,
    _phantom: PhantomData<D>,
}

impl<D: Num + Send + Sync + 'static> TriluRuntimeOp<D> {
    /// `rows` and `cols` are the input's last two extents; `k` is the diagonal offset.
    pub fn new(block_size: i32, rows: usize, cols: usize, upper: bool, k: i64) -> Self {
        let sign: i32 = if upper { 1 } else { -1 };
        // A |k| beyond either extent keeps or clears everything; clamping keeps the
        // scalar in i32 range without changing which elements are kept.
        let bound = (rows.max(cols) + 1) as i64;
        let k = k.clamp(-bound, bound) as i32;
        Self {
            block_size,
            rows: rows as i32,
            cols: cols as i32,
            sign,
            neg_sign_k: -sign * k,
            _phantom: PhantomData,
        }
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for TriluRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let n: usize = output_shape.iter().product();
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
        visitor.visit_i32(self.rows);
        visitor.visit_i32(self.cols);
        visitor.visit_i32(self.sign);
        visitor.visit_i32(self.neg_sign_k);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        [n.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}
