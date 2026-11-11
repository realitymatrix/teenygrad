//! Runs the materialisation pass over a real exported model.
//!
//! Ignored by default: it needs the FoundationStereo export, which is ~1.75 GB and
//! lives outside the repo. Point `FS_ONNX` at an `.onnx` file and run with
//! `--ignored` to see what the pass folds and what it still cannot.

use std::collections::BTreeMap;
use std::time::Instant;

use teeny_core::graph::Op;
use teeny_onnx::Onnx;
use teeny_onnx::materialize::{MaterializeConfig, materialize_constants};

#[test]
#[ignore = "needs a large external .onnx export via FS_ONNX"]
fn materialize_shrinks_a_real_export() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");

    let t0 = Instant::now();
    let (mut graph, tensors) = Onnx::from_path_with_tensors(&path).expect("import");
    println!(
        "imported {} nodes, {} tensors in {:.1}s",
        graph.nodes.len(),
        tensors.len(),
        t0.elapsed().as_secs_f64()
    );

    // Opset 13 moved Squeeze/Unsqueeze `axes` to an operand. If the importer did not
    // resolve it, every one of these nodes would carry an empty `axes`, which means a
    // different operation.
    let (mut axed, mut unaxed) = (0, 0);
    for n in &graph.nodes {
        if let Op::Squeeze { axes } | Op::Unsqueeze { axes } = &n.op {
            if axes.is_empty() {
                unaxed += 1;
            } else {
                axed += 1;
            }
        }
    }
    println!("Squeeze/Unsqueeze: {axed} with axes resolved, {unaxed} without");
    assert_eq!(unaxed, 0, "every axes operand in this export is a constant");

    let before: BTreeMap<String, usize> = tally(&graph);

    let t1 = Instant::now();
    let (folded, report) =
        materialize_constants(&mut graph, &tensors, &MaterializeConfig::default());
    println!(
        "folded {} nodes ({} values) in {:.2}s, {} skipped as too large",
        report.folded,
        folded.len(),
        t1.elapsed().as_secs_f64(),
        report.skipped_too_large
    );
    println!(
        "declined: {} Shape with unknown operand rank, {} out of range",
        report.shape_of_unknown_rank, report.declined_out_of_range
    );

    let after = tally(&graph);
    println!("\ntop unsupported ops (candidates to implement next):");
    for (op, n) in report.unsupported_ranked().iter().take(25) {
        println!("  {n:>6}  {op}");
    }

    println!("\nop counts, before -> after (changed only):");
    let mut keys: Vec<&String> = before.keys().chain(after.keys()).collect();
    keys.sort();
    keys.dedup();
    for k in keys {
        let (b, a) = (before.get(k).copied().unwrap_or(0), after.get(k).copied().unwrap_or(0));
        if b != a {
            println!("  {k:<22} {b:>6} -> {a:>6}");
        }
    }

    assert!(report.folded > 0, "nothing folded in a real export");
}

fn tally(graph: &teeny_core::graph::Graph) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for n in &graph.nodes {
        let label = format!("{:?}", n.op);
        let label = label.split([' ', '(']).next().unwrap_or("?").to_string();
        *m.entry(label).or_insert(0) += 1;
    }
    let _ = Op::Identity;
    m
}
