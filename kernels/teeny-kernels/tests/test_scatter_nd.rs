// ScatterND: a copy of one tensor with some positions overwritten.
//
// One kernel runs twice: a coordinate width of zero makes it a copy, which is how the
// output is seeded from `data`, and the real width then writes the updates. The tests
// check both passes together, since a scatter that forgets to copy looks correct
// wherever an update landed.
//
//   cargo test -p teeny-kernels --test test_scatter_nd
//   cargo test -p teeny-kernels --no-default-features --features qemu,hardware \
//       --test test_scatter_nd

use teeny_core::graph::{DtypeRepr, Graph, Op};
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::TritonLowering;

fn scatter_graph(data: &[usize], indices: &[usize], updates: &[usize]) -> Graph {
    let dims = |s: &[usize]| -> Vec<Option<usize>> { s.iter().map(|&d| Some(d)).collect() };
    let mut g = Graph::new();
    let d = g.add_node(Op::Input, vec![], DtypeRepr::F32, dims(data));
    let i = g.add_node(Op::Input, vec![], DtypeRepr::I64, dims(indices));
    let u = g.add_node(Op::Input, vec![], DtypeRepr::F32, dims(updates));
    g.add_node(Op::ScatterND, vec![d, i, u], DtypeRepr::F32, dims(data));
    g
}

fn lower(data: &[usize], indices: &[usize], updates: &[usize]) -> anyhow::Result<usize> {
    TritonLowering::new()
        .lower(&scatter_graph(data, indices, updates), LoweringMode::Inference)
        .map(|dag| dag.len())
}

#[test]
fn the_export_s_shapes_lower() {
    // A full coordinate into a rank-5 tensor, which is the export's own form.
    assert_eq!(
        lower(&[1, 8, 104, 120, 160], &[1, 8, 1, 120, 160, 5], &[1, 8, 1, 120, 160])
            .expect("rank 5"),
        4
    );
    // A partial coordinate, which scatters whole slices.
    assert_eq!(lower(&[4, 3], &[2, 1], &[2, 3]).expect("slices"), 4);
}

#[test]
fn updates_that_do_not_match_the_shapes_are_refused() {
    let msg = match lower(&[4, 3], &[2, 1], &[2, 4]) {
        Ok(_) => panic!("mismatched updates should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("call for [2, 3]"), "{msg}");
}

#[test]
fn a_coordinate_wider_than_the_rank_is_refused() {
    let msg = match lower(&[4, 3], &[2, 3], &[2]) {
        Ok(_) => panic!("a width-3 coordinate into a rank-2 tensor should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("width 3"), "{msg}");
}

/// The kernel reads the coordinates as I64, which is what ONNX gives; anything else
/// would be read at the wrong width.
#[test]
fn non_i64_indices_are_refused() {
    let mut g = Graph::new();
    let d = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(4)]);
    let i = g.add_node(Op::Input, vec![], DtypeRepr::I32, vec![Some(2), Some(1)]);
    let u = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(2)]);
    g.add_node(Op::ScatterND, vec![d, i, u], DtypeRepr::F32, vec![Some(4)]);
    let msg = match TritonLowering::new().lower(&g, LoweringMode::Inference) {
        Ok(_) => panic!("I32 indices should not lower"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("`indices` is I32"), "{msg}");
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

#[cfg(all(feature = "hardware", any(feature = "cuda", feature = "riscv")))]
mod execution {
    use super::*;
    use teeny_runtime::Target;
    use teeny_test::model::RuntimeModel;

    #[cfg(feature = "cuda")]
    fn device_and_target<'a>() -> (teeny_runtime::Device<'a>, Target) {
        let env = teeny_test::cuda::setup_cuda_env().expect("cuda env");
        let target = Target::new(env.capability);
        (env.device, target)
    }

    #[cfg(all(feature = "riscv", not(feature = "cuda")))]
    fn device_and_target<'a>() -> (teeny_runtime::Device<'a>, Target) {
        use teeny_riscv::compiler::target::Capability;
        (
            teeny_runtime::Device::new(Default::default()),
            Target::new(Capability::SpacemitK3),
        )
    }

    /// An I64 tensor's bytes as the `f32` words a model's buffers are made of: two
    /// words per element, little-endian, which is how the harness feeds anything that
    /// is not four bytes wide.
    fn i64_words(values: &[i64]) -> Vec<f32> {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
            .chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                f32::from_ne_bytes(w)
            })
            .collect()
    }

    /// ScatterND on the host: copy, then overwrite by coordinate.
    fn reference(
        data: &[f32],
        data_shape: &[usize],
        coords: &[i64],
        k: usize,
        updates: &[f32],
    ) -> Vec<f32> {
        let mut out = data.to_vec();
        let mut strides = vec![1usize; data_shape.len()];
        for ax in (0..data_shape.len().saturating_sub(1)).rev() {
            strides[ax] = strides[ax + 1] * data_shape[ax + 1];
        }
        let slice_len: usize = data_shape[k..].iter().product();
        for (row, chunk) in coords.chunks(k).enumerate() {
            let mut base = 0usize;
            for (ax, &c) in chunk.iter().enumerate() {
                let c = if c < 0 { c + data_shape[ax] as i64 } else { c };
                base += c as usize * strides[ax];
            }
            for n in 0..slice_len {
                out[base + n] = updates[row * slice_len + n];
            }
        }
        out
    }

    fn run(data_shape: &[usize], coords: &[i64], k: usize) {
        let n_rows = coords.len() / k;
        let slice_len: usize = data_shape[k..].iter().product();
        let n_data: usize = data_shape.iter().product();

        // Values distinct from the data's, so a position left uncopied or an update
        // written to the wrong place is visible either way.
        let data: Vec<f32> = (0..n_data).map(|i| i as f32 * 0.5 - 2.0).collect();
        let updates: Vec<f32> = (0..n_rows * slice_len)
            .map(|i| 1000.0 + i as f32 * 0.25)
            .collect();
        let want = reference(&data, data_shape, coords, k, &updates);

        let mut indices_shape = vec![n_rows];
        indices_shape.push(k);
        let mut updates_shape = vec![n_rows];
        updates_shape.extend_from_slice(&data_shape[k..]);

        let graph = scatter_graph(data_shape, &indices_shape, &updates_shape);
        let (device, target) = device_and_target();
        let mut model = RuntimeModel::compile(
            &device,
            &graph,
            &TritonLowering::new(),
            LoweringMode::Inference,
            &target,
            1,
            false,
        )
        .expect("compile failed");

        let idx = i64_words(coords);
        let feeds: Vec<&[f32]> = vec![&data, &idx, &updates];
        let outputs = model.terminal_node_indices_sorted_by_size();
        let got = model.run(&device, &feeds, &outputs).expect("run failed");
        let out = got
            .iter()
            .find(|o| o.len() == n_data)
            .unwrap_or_else(|| panic!("no output of length {n_data}"));
        for i in 0..n_data {
            assert_eq!(
                out[i], want[i],
                "{data_shape:?} k={k}: element {i} is {} not {}",
                out[i], want[i]
            );
        }
    }

    /// A full coordinate names one element, so the updates are scalars.
    #[test]
    #[cfg_attr(
        feature = "cuda",
        ignore = "CUDA integer division miscompiled, teenygrad#133"
    )]
    fn scattering_single_elements() {
        run(&[3, 4], &[0, 1, 2, 3, 1, 0], 2);
    }

    /// A coordinate narrower than the rank scatters whole slices, which is where a
    /// wrong slice length shows up.
    #[test]
    #[cfg_attr(
        feature = "cuda",
        ignore = "CUDA integer division miscompiled, teenygrad#133"
    )]
    fn scattering_slices() {
        run(&[4, 3], &[2, 0], 1);
    }

    /// Nothing scattered at all: the output still has to be the data, which is the
    /// pass a scatter-only kernel would get wrong.
    #[test]
    #[cfg_attr(
        feature = "cuda",
        ignore = "CUDA integer division miscompiled, teenygrad#133"
    )]
    fn scattering_nothing_still_copies() {
        run(&[2, 3], &[], 2);
    }

    /// A negative coordinate counts from the end of its own axis.
    #[test]
    #[cfg_attr(
        feature = "cuda",
        ignore = "CUDA integer division miscompiled, teenygrad#133"
    )]
    fn a_negative_coordinate_counts_from_the_end() {
        run(&[3, 4], &[-1, -1, 0, -2], 2);
    }

    /// A rank-3 tensor with both a partial coordinate and more elements than one block
    /// of lanes, so the copy pass spans several tiles.
    #[test]
    #[cfg_attr(
        feature = "cuda",
        ignore = "CUDA integer division miscompiled, teenygrad#133"
    )]
    fn a_larger_tensor_with_slices() {
        run(&[8, 5, 40], &[3, 0, 7], 1);
    }
}
