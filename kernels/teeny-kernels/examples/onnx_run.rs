//! Runs one ONNX corpus model (a directory from `scripts/onnx_corpus/export.py`) on the
//! local device and compares its outputs with onnxruntime's, for boards with no `teenyc`.
//!
//! Two modes:
//!
//! - **Ahead-of-time compile** (what `cargo teeny package` runs on the host): with
//!   `--device riscv --options capability=<chip> --cache-dir <dir> [--force]`, imports
//!   and lowers the model and compiles every kernel into `<dir>`, then exits.
//! - **Run** (on the board): `onnx_run <model_dir>`. Kernels are read from the `cache/`
//!   beside the executable, so the same lowering must have compiled them ahead of time.
//!
//! The model directory is the first positional argument, or `ONNX_RUN_MODEL` (the AOT
//! step gets only the flags above from `cargo teeny`).
//!
//! From `kernels/teeny-kernels` (cargo-teeny 0.1.3 needs a `Cross.toml` in the current
//! directory, and its `-p` does not expand this workspace's glob members):
//!
//! ```text
//! ONNX_RUN_MODEL=~/.cache/teeny-onnx/corpus/functiongemma \
//!   cargo teeny package --example onnx_run --target spacemit-k3 \
//!     --no-default-features --features riscv --dest /tmp/onnx_run_pkg
//! cargo teeny deploy --package /tmp/onnx_run_pkg --dest ~/onnx_run --host <board>
//! ssh <board> 'TEENYC_CAPABILITY=spacemit-k3 ~/onnx_run/bin/onnx_run <model_dir>'
//! ```
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use teeny_core::model::LoweringMode;
use teeny_kernels::graph::TritonLowering;
use teeny_onnx::materialize::MaterializeConfig;
use teeny_onnx::{Import, Onnx, branches::materialize_and_resolve};

struct Args {
    model: PathBuf,
    device: Option<String>,
    options: String,
    cache_dir: Option<PathBuf>,
    force: bool,
}

fn parse_args() -> Result<Args> {
    let mut model = None;
    let mut device = None;
    let mut options = String::new();
    let mut cache_dir = None;
    let mut force = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = |flag: &str| it.next().ok_or_else(|| anyhow!("{flag} needs a value"));
        match a.as_str() {
            "--device" => device = Some(value("--device")?),
            "--options" => options = value("--options")?,
            "--cache-dir" => cache_dir = Some(PathBuf::from(value("--cache-dir")?)),
            "--force" => force = true,
            s if s.starts_with("--") => bail!("unknown flag {s}"),
            s => model = Some(PathBuf::from(s)),
        }
    }
    let model = model
        .or_else(|| std::env::var_os("ONNX_RUN_MODEL").map(PathBuf::from))
        .ok_or_else(|| anyhow!("usage: onnx_run <model_dir> (or set ONNX_RUN_MODEL)"))?;
    Ok(Args {
        model,
        device,
        options,
        cache_dir,
        force,
    })
}

/// Raw bytes as the f32 words the executor's buffers and feeds are made of.
fn as_words(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            f32::from_ne_bytes(w)
        })
        .collect()
}

/// Imports and materialises the model, returning it with the lowering that compiles it.
/// The AOT step and the run must build the same lowering, so both come through here.
fn prepare(model_path: &Path) -> Result<(Import, TritonLowering)> {
    let mut import = Onnx::import_from_path(model_path)
        .with_context(|| format!("importing {}", model_path.display()))?;
    let (folded, _, _) = materialize_and_resolve(&mut import, &MaterializeConfig::default())?;
    let mut constants = import.tensors.clone();
    constants.extend(folded);
    let lowering = TritonLowering::new().with_constants(constants);
    Ok((import, lowering))
}

#[cfg(feature = "riscv")]
fn aot(args: &Args, device: &str) -> Result<()> {
    ensure!(
        device == "riscv",
        "this example AOT-compiles only for --device riscv, not '{device}'"
    );
    let (import, lowering) = prepare(&args.model.join("model.onnx"))?;
    let options = teeny_riscv::compiler::options::Options::parse(&args.options)?;
    let cache_dir = args
        .cache_dir
        .clone()
        .or_else(|| std::env::var_os("TEENYC_CACHE_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/tmp/teenyc_cache"));
    std::fs::create_dir_all(&cache_dir)?;
    let t = Instant::now();
    let kernels = teeny_riscv::compiler::aot::compile_graph(
        &import.graph,
        &lowering,
        LoweringMode::Inference,
        &options,
        cache_dir.to_string_lossy().as_ref(),
        args.force,
    )?;
    println!(
        "compiled {} kernels for {:?} into {} in {:.1} s",
        kernels.len(),
        options.capability,
        cache_dir.display(),
        t.elapsed().as_secs_f32()
    );
    Ok(())
}

#[cfg(not(feature = "riscv"))]
fn aot(_args: &Args, device: &str) -> Result<()> {
    bail!("--device {device}: build with --features riscv")
}

fn run(dir: &Path) -> Result<()> {
    use teeny_test::model::RuntimeModel;

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let list = |key: &str| -> Result<Vec<(String, String, Vec<u8>)>> {
        manifest[key]
            .as_array()
            .ok_or_else(|| anyhow!("manifest has no '{key}' list"))?
            .iter()
            .map(|t| {
                let file = dir.join(t["file"].as_str().unwrap_or_default());
                let bytes = std::fs::read(&file).with_context(|| file.display().to_string())?;
                Ok((
                    t["name"].as_str().unwrap_or_default().to_string(),
                    t["dtype"].as_str().unwrap_or("f32").to_string(),
                    bytes,
                ))
            })
            .collect()
    };
    let inputs: BTreeMap<String, Vec<u8>> = list("inputs")?
        .into_iter()
        .map(|(n, _, b)| (n, b))
        .collect();
    let outputs = list("outputs")?;

    let t = Instant::now();
    let (import, lowering) = prepare(&dir.join("model.onnx"))?;
    let (dag, graph_to_dag) =
        lowering.lower_with_mapping(&import.graph, LoweringMode::Inference)?;
    let elided = teeny_kernels::graph::elidable_constants(&import.graph);
    let mut dag_to_graph = vec![usize::MAX; dag.len()];
    for (g, &d) in graph_to_dag.iter().enumerate() {
        if d < dag.len() && !elided[g] {
            dag_to_graph[d] = g;
        }
    }
    let mut feeds: Vec<Vec<f32>> = Vec::new();
    for d in dag.topological_sort() {
        if !dag.node(d).value.as_ref().is_input() {
            continue;
        }
        let g = dag_to_graph[d];
        let name = import.graph.names.get(&g).cloned().unwrap_or_default();
        if let Some(bytes) = inputs.get(&name) {
            feeds.push(as_words(bytes));
        } else if let Some(t) = import.tensors.get(&g) {
            feeds.push(as_words(&t.bytes));
        } else {
            bail!("Input node {g} ('{name}') is neither a model input nor a constant");
        }
    }
    println!("import + lower: {:.1} s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    let device = teeny_runtime::open()?;
    let target = teeny_runtime::default_target(&device)?;
    let mut model = RuntimeModel::compile(
        &device,
        &import.graph,
        &lowering,
        LoweringMode::Inference,
        &target,
        1,
        false,
    )?;
    let mut by_node: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (slot, node, idx) in model
        .param_slots()
        .map(|(s, n, i)| (format!(".{s}"), n, i))
        .collect::<Vec<_>>()
    {
        if [".weight", ".bias", ".running_mean", ".running_var"]
            .iter()
            .any(|w| slot.ends_with(w))
        {
            by_node.entry(node).or_default().push(idx);
        }
    }
    for (node, mut params) in by_node {
        params.sort_unstable();
        let g = dag_to_graph[node];
        if g == usize::MAX {
            continue;
        }
        let operands: Vec<usize> = import.graph.nodes[g]
            .inputs
            .iter()
            .copied()
            .filter(|i| import.tensors.contains_key(i))
            .collect();
        let sources: Vec<usize> = if operands.is_empty() {
            import
                .tensors
                .contains_key(&g)
                .then_some(g)
                .into_iter()
                .collect()
        } else {
            operands
        };
        for (idx, src) in params.into_iter().zip(sources) {
            model.load_param_f32(node, idx, &as_words(&import.tensors[&src].bytes))?;
        }
    }
    println!("load kernels + weights: {:.1} s", t.elapsed().as_secs_f32());

    let wanted: Vec<usize> = outputs
        .iter()
        .map(|(name, _, _)| {
            let g = *import
                .value_to_node
                .get(name)
                .ok_or_else(|| anyhow!("no node produces output '{name}'"))?;
            let d = graph_to_dag[g];
            ensure!(
                d < dag.len() && !elided[g],
                "output '{name}' was lowered away"
            );
            Ok(d)
        })
        .collect::<Result<_>>()?;
    let refs: Vec<&[f32]> = feeds.iter().map(Vec::as_slice).collect();
    let t = Instant::now();
    let got = model.run(&device, &refs, &wanted)?;
    println!("run: {:.2} s", t.elapsed().as_secs_f32());

    let mut ok = true;
    for ((name, dtype, want), got) in outputs.iter().zip(&got) {
        ensure!(
            dtype == "f32",
            "output '{name}' is {dtype}; only f32 outputs are compared"
        );
        let want: Vec<f32> = want
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        ensure!(
            got.len() >= want.len(),
            "output '{name}': {} values, want {}",
            got.len(),
            want.len()
        );
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = got.iter().zip(&want).fold(0.0f32, |m, (g, w)| {
            if g.is_finite() {
                m.max((g - w).abs() / scale)
            } else {
                f32::INFINITY
            }
        });
        let pass = worst <= 1e-3;
        ok &= pass;
        println!(
            "{name}: {} worst rel. diff {worst:.1e}",
            if pass { "PASS" } else { "FAIL" }
        );
    }
    if !ok {
        bail!("outputs differ from onnxruntime");
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    match args.device.clone() {
        Some(device) => aot(&args, &device),
        None => run(&args.model),
    }
}
