// strided_copy_forward: one kernel for Transpose, Slice and Expand.
//
// Each case is checked against NumPy's own result for the same operation, so the
// index mapping is validated rather than merely self-consistent.

#![cfg(feature = "cuda")]

use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::model::RuntimeOp;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_kernels::nn::tensor::strided_copy::{StridedCopyForward, StridedCopyRuntimeOp};
use teeny_test::cuda as testing;

const BLOCK: i32 = 128;

fn bin(name: &str) -> Vec<f32> {
    let p = format!("/tmp/strided/{name}.bin");
    std::fs::read(&p)
        .unwrap_or_else(|e| panic!("missing {p}: {e}"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Runs the kernel for `op` and compares against NumPy's output in `want`.
fn check(
    label: &str,
    input: &[f32],
    want: &[f32],
    out_shape: &[usize],
    op: StridedCopyRuntimeOp<f32>,
) -> Result<()> {
    dotenv::dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let mut x_buf = device.buffer::<f32>(input.len())?;
    let out_buf = device.buffer::<f32>(want.len())?;
    x_buf.to_device(input)?;

    let kernel = StridedCopyForward::<f32>::new(BLOCK);
    let ptx = std::fs::read(compile_kernel(
        &kernel,
        &Target::new(env.capability),
        true,
        false,
    )?)?;
    let program = testing::load_program_from_ptx::<StridedCopyForward<f32>>(&ptx)?;

    // Scalars come from the RuntimeOp, so the test exercises the same mapping the
    // lowering will use rather than a hand-written duplicate.
    let n: usize = out_shape.iter().product();
    let a = op.kernel_scalars(n as i32);

    let grid = op.grid(out_shape);
    let cfg = testing::launch_config_with_grid(grid[0] as usize, &program);
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            out_buf.as_device_ptr(),
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            a[5],
            a[6],
            a[7],
            a[8],
            a[9],
            a[10],
            a[11],
            a[12],
            a[13],
        ),
    )?;

    let mut got = vec![0.0f32; want.len()];
    out_buf.to_host(&mut got)?;
    let mut max = 0.0f32;
    for i in 0..want.len() {
        let d = (got[i] - want[i]).abs();
        if d > max {
            max = d;
        }
    }
    println!("{label}: max |kernel - numpy| = {max:e}");
    assert!(max == 0.0, "{label} mismatch: {max:e}");
    Ok(())
}

#[test]
fn test_strided_copy_transpose() -> Result<()> {
    let x = bin("x");
    check(
        "transpose (2,3,4,5)->(0,2,3,1)",
        &x,
        &bin("transpose"),
        &[2, 4, 5, 3],
        StridedCopyRuntimeOp::<f32>::transpose(BLOCK, &[2, 3, 4, 5], &[0, 2, 3, 1]),
    )
}

#[test]
fn test_strided_copy_slice() -> Result<()> {
    let x = bin("x");
    check(
        "slice [:,1:3,0:4:2,1:5:3]",
        &x,
        &bin("slice"),
        &[2, 2, 2, 2],
        StridedCopyRuntimeOp::<f32>::slice(
            BLOCK,
            &[2, 3, 4, 5],
            &[2, 2, 2, 2],
            &[0, 1, 0, 1],
            &[1, 1, 2, 3],
        ),
    )
}

#[test]
fn test_strided_copy_expand() -> Result<()> {
    let src = bin("expand_src");
    check(
        "expand (1,3,1,5)->(2,3,4,5)",
        &src,
        &bin("expand"),
        &[2, 3, 4, 5],
        StridedCopyRuntimeOp::<f32>::expand(BLOCK, &[1, 3, 1, 5], &[2, 3, 4, 5]),
    )
}
