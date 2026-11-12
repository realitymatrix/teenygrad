//! How far materialised constants get the FoundationStereo graph through lowering.
//!
//! Ignored by default: needs the export, which is ~1.75 GB and lives outside the
//! repo. Point `FS_ONNX` at an `.onnx` file and run with `--ignored`.

use teeny_core::graph::Op;
use teeny_core::model::{Lowering, LoweringMode};
use teeny_kernels::graph::{TritonLowering, elidable_constants};
use teeny_onnx::Onnx;
use teeny_onnx::branches::materialize_and_resolve;
use teeny_onnx::materialize::MaterializeConfig;

#[test]
#[ignore = "needs a large external .onnx export via FS_ONNX"]
fn materialised_constants_clear_the_shape_operands() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to an .onnx path");
    let mut import = Onnx::import_from_path(&path).expect("import");
    println!("unresolved If nodes at import: {}", import.branches.len());
    // Before materialisation, every shape operand is a live op, so nothing is
    // elidable and lowering has to produce a buffer for each.
    let before = count_constants(&import.graph, &elidable_constants(&import.graph));
    println!("before materialisation: {before:?}");

    // One call does the folding and the branch resolution, keeping the constant map
    // seeded correctly across rounds. Materialising separately here as well would
    // mis-seed this call and silently lose the first round's folds.
    let (all, report, branch_report) =
        materialize_and_resolve(&mut import, &MaterializeConfig::default()).expect("resolve");
    println!("folded {} nodes (first round)", report.folded);
    println!("branches: {branch_report:?}");

    let graph = &import.graph;
    let elide = elidable_constants(graph);
    let after = count_constants(graph, &elide);
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

    println!("constant values available to lowering: {}", all.len());

    // Gather's indices are a tensor operand, not shape metadata. Whether they are
    // constant decides what a Gather kernel has to support, so count both.
    {
        let mut const_idx = 0usize;
        let mut dynamic_idx = 0usize;
        let mut idx_lens: std::collections::BTreeMap<String, usize> = Default::default();
        for node in &graph.nodes {
            if !matches!(node.op, Op::Gather { .. }) {
                continue;
            }
            match node.inputs.get(1) {
                Some(&i) if all.contains_key(&i) => {
                    const_idx += 1;
                    let n = all[&i].len().unwrap_or(0);
                    let bucket = match n {
                        0 => "0",
                        1 => "1 (scalar)",
                        2..=16 => "2-16",
                        _ => "17+",
                    };
                    *idx_lens.entry(bucket.to_string()).or_insert(0) += 1;
                }
                _ => dynamic_idx += 1,
            }
        }
        println!(
            "Gather nodes: {const_idx} with materialised indices, {dynamic_idx} dynamic"
        );
        println!("  constant index counts: {idx_lens:?}");
    }

    // What ops lowering actually has to handle, most common first. The lowering error
    // names an op family rather than a node, so without this the next blocker is a
    // guess: an op's *presence* in the error is not evidence of how much of the model
    // depends on it.
    {
        let mut by_op: std::collections::BTreeMap<String, usize> = Default::default();
        for (i, node) in graph.nodes.iter().enumerate() {
            if elide[i] {
                continue;
            }
            let dbg = format!("{:?}", node.op);
            let variant = dbg
                .split_once([' ', '(', '{'])
                .map_or(dbg.as_str(), |(head, _)| head)
                .to_string();
            *by_op.entry(variant).or_insert(0) += 1;
        }
        let mut ranked: Vec<_> = by_op.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        println!("ops lowering must handle ({} kinds):", ranked.len());
        for (op, n) in &ranked {
            println!("  {n:>6}  {op}");
        }
    }

    // Attempt lowering and report where it stops, so Op::Constant's absence from the
    // message is visible rather than assumed.
    let lowering = TritonLowering::new().with_constants(all);
    match lowering.lower(graph, LoweringMode::Inference) {
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
