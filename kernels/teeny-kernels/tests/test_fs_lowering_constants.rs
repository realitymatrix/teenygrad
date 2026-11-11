//! How far materialised constants get the FoundationStereo graph through lowering.
//!
//! Ignored by default: needs the export, which is ~1.75 GB and lives outside the
//! repo. Point `FS_ONNX` at an `.onnx` file and run with `--ignored`.

use teeny_core::graph::Op;
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::{TritonLowering, elidable_constants};
use teeny_onnx::Onnx;
use teeny_onnx::materialize::{MaterializeConfig, materialize_constants};

#[test]
#[ignore = "needs a large external .onnx export via FS_ONNX"]
fn materialised_constants_clear_the_shape_operands() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let (mut graph, tensors) = Onnx::from_path_with_tensors(&path).expect("import");

    // Before materialisation, every shape operand is a live op, so nothing is
    // elidable and lowering has to produce a buffer for each.
    let before = count_constants(&graph, &elidable_constants(&graph));
    println!("before materialisation: {before:?}");

    let (folded, report) =
        materialize_constants(&mut graph, &tensors, &MaterializeConfig::default());
    println!("folded {} nodes", report.folded);

    let elide = elidable_constants(&graph);
    let after = count_constants(&graph, &elide);
    println!("after materialisation:  {after:?}");

    assert!(
        after.elidable > 0,
        "materialisation should make shape operands elidable"
    );

    // What would still need a device buffer, by dtype: these are the data constants,
    // the remaining work for lowering.
    let mut by_dtype: std::collections::BTreeMap<String, usize> = Default::default();
    for (i, node) in graph.nodes.iter().enumerate() {
        if matches!(node.op, Op::Constant { .. }) && !elide[i] {
            *by_dtype.entry(format!("{:?}", node.dtype)).or_insert(0) += 1;
        }
    }
    println!("data constants still needing a buffer, by dtype: {by_dtype:?}");

    // Attempt lowering and report where it stops, so Op::Constant's absence from the
    // message is visible rather than assumed.
    let lowering = TritonLowering::new().with_constants(folded);
    match lowering.lower(&graph, LoweringMode::Inference) {
        Ok(dag) => println!("lowered the whole graph: {} DAG nodes", dag.len()),
        Err(e) => println!("lowering stops at: {e}"),
    }
}

#[derive(Debug)]
struct Counts {
    constants: usize,
    elidable: usize,
}

fn count_constants(graph: &teeny_core::graph::Graph, elide: &[bool]) -> Counts {
    let mut c = Counts {
        constants: 0,
        elidable: 0,
    };
    for (i, node) in graph.nodes.iter().enumerate() {
        if matches!(node.op, Op::Constant { .. }) {
            c.constants += 1;
            if elide[i] {
                c.elidable += 1;
            }
        }
    }
    c
}
