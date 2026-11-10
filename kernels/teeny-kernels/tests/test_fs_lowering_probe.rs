// Reports how far FoundationStereo's ONNX graph gets through lowering.
//
// Imports the model, lowers it, and prints the first op with no lowering — so progress
// is measured against the real model rather than a count of match arms.
//
//   FS_ONNX=<path> cargo test -p teeny-kernels --test test_fs_lowering_probe \
//       -- --nocapture --ignored

use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;

#[test]
#[ignore = "needs FS_ONNX pointing at a FoundationStereo .onnx export"]
fn probe_fs_lowering() {
    let path = std::env::var("FS_ONNX").expect("set FS_ONNX to a FoundationStereo .onnx");
    let graph = teeny_onnx::Onnx::from_path(&path).expect("import failed");
    println!("imported {} graph nodes", graph.nodes.len());

    let lowering = TritonLowering::new();
    match lowering.lower_with_mapping(&graph, LoweringMode::Inference) {
        Ok((dag, _)) => println!("LOWERED OK — {} dag nodes", dag.len()),
        Err(e) => println!("first blocker: {e}"),
    }
}
