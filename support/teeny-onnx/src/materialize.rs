//! Constant and shape materialisation for imported ONNX graphs.
//!
//! Exported models carry two very different kinds of constant. A handful are real
//! weights, which can run to gigabytes and must stay where they are. The rest is
//! *shape plumbing*: thousands of tiny `Shape`/`Gather`/`Concat`/`Unsqueeze` chains
//! that an exporter emits to compute a reshape target or a broadcast size. Those
//! chains depend on nothing but the model itself, so they have a single answer, but
//! a lowering pass that meets them one node at a time cannot see it — it only sees
//! ops whose operands are not known until "runtime".
//!
//! This pass evaluates those chains ahead of lowering and rewrites each one to an
//! [`Op::Constant`] holding its value, which is what `Op::Constant`'s own
//! documentation asks for. Two properties keep it cheap:
//!
//! * It works entirely in memory on a [`Graph`], so a model that cannot be written
//!   back out through protobuf's 2 GB ceiling is not a problem.
//! * It only ever decodes tensors of at most [`MaterializeConfig::max_elements`]
//!   elements, so the weights are referenced but never touched. Folding stays
//!   proportional to the shape plumbing, not to the model size.
//!
//! Anything it cannot evaluate it leaves exactly as it was and counts in
//! [`MaterializeReport::unsupported`], so the pass is always safe to run and the
//! report says what the next op worth supporting is.

use std::collections::BTreeMap;

use teeny_core::graph::{DtypeRepr, Graph, Op, Shape};

use crate::onnx::TensorData;

/// Default element ceiling for a foldable tensor.
///
/// Shape plumbing deals in rank-0 and rank-1 tensors of a few dozen elements. The
/// default leaves generous headroom for things like a flattened position-embedding
/// index while staying far below any real weight.
pub const DEFAULT_MAX_ELEMENTS: usize = 65_536;

/// Tuning for [`materialize_constants`].
#[derive(Debug, Clone)]
pub struct MaterializeConfig {
    /// Largest tensor, in elements, the pass will decode or produce. A node whose
    /// value would exceed this is left alone and counted in
    /// [`MaterializeReport::skipped_too_large`].
    pub max_elements: usize,
}

impl Default for MaterializeConfig {
    fn default() -> Self {
        Self {
            max_elements: DEFAULT_MAX_ELEMENTS,
        }
    }
}

/// What [`materialize_constants`] did.
#[derive(Debug, Default, Clone)]
pub struct MaterializeReport {
    /// Nodes rewritten to [`Op::Constant`].
    pub folded: usize,
    /// Nodes left alone because their value would exceed the element ceiling.
    pub skipped_too_large: usize,
    /// Nodes left alone because the pass cannot evaluate the op, keyed by a
    /// human-readable op name with the number of occurrences.
    pub unsupported: BTreeMap<String, usize>,
    /// `Shape` nodes declined because the operand's rank is not known.
    ///
    /// [`Shape`] cannot tell a genuine rank-0 tensor from a tensor whose shape the
    /// importer never inferred: both are the empty vector. Folding the latter would
    /// silently produce an empty shape vector and corrupt everything downstream, so
    /// the pass refuses. A non-zero count here means the importer's shape coverage,
    /// not this pass, is what limits how much folds.
    pub shape_of_unknown_rank: usize,
    /// Nodes declined because an index or operand was out of range for the shapes
    /// the pass was given.
    ///
    /// On a model whose shapes are fully inferred this should be zero. A non-zero
    /// count means some folded value disagrees with a consumer about a tensor's
    /// real shape, which again points at importer shape coverage.
    pub declined_out_of_range: usize,
}

impl MaterializeReport {
    /// The unsupported ops, most frequent first. Useful for deciding what to
    /// implement next.
    pub fn unsupported_ranked(&self) -> Vec<(&str, usize)> {
        let mut v: Vec<(&str, usize)> = self
            .unsupported
            .iter()
            .map(|(k, n)| (k.as_str(), *n))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
    }
}

/// Evaluates every foldable constant/shape subgraph in `graph` and rewrites it in
/// place to an [`Op::Constant`].
///
/// `tensors` is the value map from [`crate::Onnx::from_path_with_tensors`]: it seeds
/// the pass with initializer and `Constant` values. Tensors above the element
/// ceiling are skipped rather than decoded, so passing the full map of a multi-
/// gigabyte model is safe.
///
/// Returns the values of the newly folded nodes, keyed by node index, alongside the
/// report. `Op::Input` nodes are never rewritten: an initializer stays an input so
/// its weights can still be bound by name.
pub fn materialize_constants(
    graph: &mut Graph,
    tensors: &BTreeMap<usize, TensorData>,
    config: &MaterializeConfig,
) -> (BTreeMap<usize, TensorData>, MaterializeReport) {
    let mut report = MaterializeReport::default();
    let mut values: BTreeMap<usize, Value> = BTreeMap::new();

    // Seed from the model's own tensors, skipping anything large. This is the step
    // that keeps the pass away from the weights.
    //
    // A `ConstantOfShape`'s seeded tensor is its scalar *fill value*, not its
    // output, so it goes in a separate table. Seeding it as a value would make the
    // node look already-known, and a scalar would stand in for the whole tensor.
    let mut fills: BTreeMap<usize, Value> = BTreeMap::new();
    for (&id, td) in tensors {
        if elem_count(&td.shape).is_none_or(|n| n > config.max_elements) {
            continue;
        }
        let Some(v) = Value::decode(td) else {
            continue;
        };
        if matches!(graph.nodes.get(id).map(|n| &n.op), Some(Op::ConstantOfShape { .. })) {
            fills.insert(id, v);
        } else {
            values.insert(id, v);
        }
    }

    let order = graph.topological_sort();
    let mut folded: BTreeMap<usize, TensorData> = BTreeMap::new();

    for id in order {
        if values.contains_key(&id) {
            // Already a known constant (an initializer or a `Constant` node). Only
            // record a rewrite for `Constant`, whose value now lives in the op.
            continue;
        }

        let node = &graph.nodes[id];
        if matches!(node.op, Op::Input) {
            continue;
        }

        let input_shapes: Vec<&Shape> = node
            .inputs
            .iter()
            .map(|&i| &graph.nodes[i].shape)
            .collect();

        // `Shape` reads its operand's *shape*, not its value, so it folds over a
        // live activation. It is the root of nearly every plumbing chain, so this
        // case has to come before the all-operands-known test below.
        let args: Vec<&Value> = if matches!(node.op, Op::ShapeOf { .. }) {
            Vec::new()
        } else {
            // Every other op needs each operand's value to have a single answer.
            match node
                .inputs
                .iter()
                .map(|i| values.get(i))
                .collect::<Option<Vec<&Value>>>()
            {
                Some(args) => args,
                None => continue,
            }
        };

        let fill = fills.get(&id).filter(|f| f.len() > 0);
        match eval(&node.op, &args, &input_shapes, fill, config) {
            Eval::Ok(v) => {
                if v.len() > config.max_elements {
                    report.skipped_too_large += 1;
                    continue;
                }
                let td = v.encode();
                let shape: Shape = v.shape.iter().map(|&d| Some(d)).collect();
                let dtype = v.dtype;
                values.insert(id, v);
                folded.insert(id, td);

                let node = &mut graph.nodes[id];
                node.op = Op::Constant { dtype, shape: shape.clone() };
                node.inputs.clear();
                node.dtype = dtype;
                node.shape = shape;
                report.folded += 1;
            }
            Eval::TooLarge => report.skipped_too_large += 1,
            Eval::UnknownRank => report.shape_of_unknown_rank += 1,
            Eval::OutOfRange => report.declined_out_of_range += 1,
            Eval::Unsupported => {
                *report
                    .unsupported
                    .entry(op_label(&node.op))
                    .or_insert(0) += 1;
            }
        }
    }

    (folded, report)
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// A materialised tensor, held in whichever of three lanes its dtype belongs to.
///
/// Integers use `i64` and floats `f64` so that every ONNX dtype the pass meets
/// round-trips without loss: shape arithmetic is exact in `i64`, and `f64` holds
/// every `f32` exactly.
#[derive(Debug, Clone, PartialEq)]
struct Value {
    dtype: DtypeRepr,
    shape: Vec<usize>,
    data: Data,
}

#[derive(Debug, Clone, PartialEq)]
enum Data {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Bool(Vec<bool>),
}

/// The outcome of trying to evaluate one node.
enum Eval {
    Ok(Value),
    /// Evaluable, but the result would exceed the element ceiling.
    TooLarge,
    /// The pass does not implement this op, or an operand combination it needs.
    Unsupported,
    /// A `Shape` whose operand's rank the IR does not pin down.
    UnknownRank,
    /// Evaluable in principle, but an index or operand was out of range.
    OutOfRange,
}

impl Value {
    fn len(&self) -> usize {
        match &self.data {
            Data::Int(v) => v.len(),
            Data::Float(v) => v.len(),
            Data::Bool(v) => v.len(),
        }
    }

    fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Reads element `i` as an `i64`, for shape arithmetic. Floats truncate.
    fn int_at(&self, i: usize) -> i64 {
        match &self.data {
            Data::Int(v) => v[i],
            Data::Float(v) => v[i] as i64,
            Data::Bool(v) => v[i] as i64,
        }
    }

    fn float_at(&self, i: usize) -> f64 {
        match &self.data {
            Data::Int(v) => v[i] as f64,
            Data::Float(v) => v[i],
            Data::Bool(v) => v[i] as i64 as f64,
        }
    }

    fn bool_at(&self, i: usize) -> bool {
        match &self.data {
            Data::Int(v) => v[i] != 0,
            Data::Float(v) => v[i] != 0.0,
            Data::Bool(v) => v[i],
        }
    }

    /// The whole tensor as `i64`s. Shape vectors arrive this way.
    fn as_ints(&self) -> Vec<i64> {
        (0..self.len()).map(|i| self.int_at(i)).collect()
    }

    fn is_float(&self) -> bool {
        matches!(self.data, Data::Float(_))
    }

    /// Rebuilds a value with the same dtype and lane but a new shape, keeping the
    /// element order. Used by the reshaping ops, which never move data.
    fn with_shape(&self, shape: Vec<usize>) -> Value {
        Value {
            dtype: self.dtype,
            shape,
            data: self.data.clone(),
        }
    }

    /// Picks the listed flat source indices out of this value, into `shape`.
    fn gather_flat(&self, idx: &[usize], shape: Vec<usize>) -> Value {
        let data = match &self.data {
            Data::Int(v) => Data::Int(idx.iter().map(|&i| v[i]).collect()),
            Data::Float(v) => Data::Float(idx.iter().map(|&i| v[i]).collect()),
            Data::Bool(v) => Data::Bool(idx.iter().map(|&i| v[i]).collect()),
        };
        Value {
            dtype: self.dtype,
            shape,
            data,
        }
    }

    fn decode(td: &TensorData) -> Option<Value> {
        let shape: Vec<usize> = td.shape.iter().copied().collect::<Option<Vec<usize>>>()?;
        let n = shape.iter().product::<usize>();
        let b = &td.bytes;

        macro_rules! ints {
            ($ty:ty, $w:expr) => {{
                if b.len() < n * $w {
                    return None;
                }
                Data::Int(
                    b.as_chunks::<$w>()
                        .0
                        .iter()
                        .take(n)
                        .map(|c| <$ty>::from_le_bytes(*c) as i64)
                        .collect(),
                )
            }};
        }

        let data = match td.dtype {
            DtypeRepr::Bool => {
                if b.len() < n {
                    return None;
                }
                Data::Bool(b.iter().take(n).map(|&x| x != 0).collect())
            }
            DtypeRepr::I8 => ints!(i8, 1),
            DtypeRepr::U8 => ints!(u8, 1),
            DtypeRepr::I16 => ints!(i16, 2),
            DtypeRepr::U16 => ints!(u16, 2),
            DtypeRepr::I32 => ints!(i32, 4),
            DtypeRepr::U32 => ints!(u32, 4),
            DtypeRepr::I64 => ints!(i64, 8),
            DtypeRepr::U64 => {
                if b.len() < n * 8 {
                    return None;
                }
                Data::Int(
                    b.as_chunks::<8>()
                        .0
                        .iter()
                        .take(n)
                        .map(|c| u64::from_le_bytes(*c) as i64)
                        .collect(),
                )
            }
            DtypeRepr::F32 => {
                if b.len() < n * 4 {
                    return None;
                }
                Data::Float(
                    b.as_chunks::<4>()
                        .0
                        .iter()
                        .take(n)
                        .map(|c| f32::from_le_bytes(*c) as f64)
                        .collect(),
                )
            }
            DtypeRepr::F64 => {
                if b.len() < n * 8 {
                    return None;
                }
                Data::Float(
                    b.as_chunks::<8>()
                        .0
                        .iter()
                        .take(n)
                        .map(|c| f64::from_le_bytes(*c))
                        .collect(),
                )
            }
            // 16-bit floats have no native Rust type here. Shape plumbing never uses
            // them, so leaving them unfolded costs nothing.
            DtypeRepr::F16 | DtypeRepr::BF16 => return None,
        };

        Some(Value {
            dtype: td.dtype,
            shape,
            data,
        })
    }

    fn encode(&self) -> TensorData {
        let mut bytes = Vec::new();
        match (&self.data, self.dtype) {
            (_, DtypeRepr::Bool) => {
                for i in 0..self.len() {
                    bytes.push(self.bool_at(i) as u8);
                }
            }
            (_, DtypeRepr::I8) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as i8).to_le_bytes());
                }
            }
            (_, DtypeRepr::U8) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as u8).to_le_bytes());
                }
            }
            (_, DtypeRepr::I16) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as i16).to_le_bytes());
                }
            }
            (_, DtypeRepr::U16) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as u16).to_le_bytes());
                }
            }
            (_, DtypeRepr::I32) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as i32).to_le_bytes());
                }
            }
            (_, DtypeRepr::U32) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as u32).to_le_bytes());
                }
            }
            (_, DtypeRepr::I64) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&self.int_at(i).to_le_bytes());
                }
            }
            (_, DtypeRepr::U64) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.int_at(i) as u64).to_le_bytes());
                }
            }
            (_, DtypeRepr::F32) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&(self.float_at(i) as f32).to_le_bytes());
                }
            }
            (_, DtypeRepr::F64) => {
                for i in 0..self.len() {
                    bytes.extend_from_slice(&self.float_at(i).to_le_bytes());
                }
            }
            (_, DtypeRepr::F16 | DtypeRepr::BF16) => {}
        }
        TensorData {
            dtype: self.dtype,
            shape: self.shape.iter().map(|&d| Some(d)).collect(),
            bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

fn elem_count(shape: &Shape) -> Option<usize> {
    shape
        .iter()
        .copied()
        .collect::<Option<Vec<usize>>>()
        .map(|d| d.iter().product())
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Right-aligned numpy/ONNX broadcasting of two shapes.
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = dim_from_right(a, rank, i);
        let db = dim_from_right(b, rank, i);
        out.push(match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        });
    }
    Some(out)
}

fn dim_from_right(shape: &[usize], rank: usize, i: usize) -> usize {
    let pad = rank - shape.len();
    if i < pad { 1 } else { shape[i - pad] }
}

/// Maps a coordinate in the broadcast output back to a flat index in `shape`.
fn broadcast_offset(coords: &[usize], shape: &[usize]) -> usize {
    let rank = coords.len();
    let strides = contiguous_strides(shape);
    let pad = rank - shape.len();
    let mut off = 0;
    for (ax, &d) in shape.iter().enumerate() {
        // A size-1 axis is stretched: every output coordinate reads index 0.
        if d != 1 {
            off += coords[ax + pad] * strides[ax];
        }
    }
    off
}

fn coords_of(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let strides = contiguous_strides(shape);
    let mut c = vec![0usize; shape.len()];
    for ax in 0..shape.len() {
        c[ax] = flat / strides[ax];
        flat -= c[ax] * strides[ax];
    }
    c
}

/// Normalises a possibly-negative axis against `rank`.
fn norm_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    (0..=r).contains(&a).then_some(a as usize)
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

fn eval(
    op: &Op,
    args: &[&Value],
    input_shapes: &[&Shape],
    fill: Option<&Value>,
    config: &MaterializeConfig,
) -> Eval {
    match op {
        // --- plumbing sources ---
        Op::ShapeOf { start, end } => eval_shape_of(*start, *end, input_shapes),
        Op::ConstantOfShape { dtype } => eval_constant_of_shape(*dtype, args, fill, config),
        Op::Range => eval_range(args, config),

        // --- pure reshapes: element order never changes ---
        Op::Identity => one(args).map_or(Eval::Unsupported, |v| Eval::Ok(v.clone())),
        Op::Reshape => eval_reshape(args),
        Op::Squeeze { axes } => eval_squeeze(args, axes),
        Op::Unsqueeze { axes } => eval_unsqueeze(args, axes),

        // --- data movement ---
        Op::Concat { axis } => eval_concat(args, *axis, config),
        Op::Gather { axis } => eval_gather(args, *axis, config),
        Op::Slice => eval_slice(args),
        Op::Transpose { perm } => eval_transpose(args, perm),
        Op::Expand => eval_expand(args, config),

        // --- arithmetic and comparison ---
        Op::Add => binary(args, BinKind::Add),
        Op::Sub => binary(args, BinKind::Sub),
        Op::Mul => binary(args, BinKind::Mul),
        Op::Div => binary(args, BinKind::Div),
        Op::Pow => binary(args, BinKind::Pow),
        Op::Mod { fmod } => binary(args, BinKind::Mod { fmod: *fmod }),
        Op::Equal => binary(args, BinKind::Equal),
        Op::Greater => binary(args, BinKind::Greater),
        Op::GreaterOrEqual => binary(args, BinKind::GreaterOrEqual),
        Op::Less => binary(args, BinKind::Less),
        Op::LessOrEqual => binary(args, BinKind::LessOrEqual),
        Op::Where => eval_where(args),
        Op::Neg => unary(args, UnKind::Neg),
        Op::Abs => unary(args, UnKind::Abs),
        Op::Sqrt => unary(args, UnKind::Sqrt),
        Op::Floor => unary(args, UnKind::Floor),
        Op::Ceil => unary(args, UnKind::Ceil),
        Op::Not => unary(args, UnKind::Not),
        Op::Cast { to } => eval_cast(args, *to),

        _ => Eval::Unsupported,
    }
}

fn one<'a>(args: &[&'a Value]) -> Option<&'a Value> {
    (args.len() == 1).then(|| args[0])
}

fn eval_shape_of(start: i64, end: i64, input_shapes: &[&Shape]) -> Eval {
    let Some(shape) = input_shapes.first() else {
        return Eval::Unsupported;
    };
    if shape.is_empty() {
        // Ambiguous: either a real rank-0 tensor or a shape the importer never
        // inferred. See `MaterializeReport::shape_of_unknown_rank`.
        return Eval::UnknownRank;
    }
    let rank = shape.len() as i64;
    // ONNX clamps both ends into range; `end` defaults to the rank.
    let lo = (if start < 0 { start + rank } else { start }).clamp(0, rank) as usize;
    let hi = (if end < 0 { end + rank } else { end }).clamp(0, rank) as usize;
    if lo > hi {
        return Eval::Unsupported;
    }
    // A dynamic dim has no single answer, so the whole `Shape` stays unfolded.
    let Some(dims) = shape[lo..hi].iter().copied().collect::<Option<Vec<usize>>>() else {
        return Eval::Unsupported;
    };
    Eval::Ok(Value {
        dtype: DtypeRepr::I64,
        shape: vec![dims.len()],
        data: Data::Int(dims.iter().map(|&d| d as i64).collect()),
    })
}

fn eval_constant_of_shape(
    dtype: DtypeRepr,
    args: &[&Value],
    fill: Option<&Value>,
    config: &MaterializeConfig,
) -> Eval {
    // args[0] is the target shape. `fill` is the node's `value` attribute, a scalar;
    // ONNX defaults it to zero when the attribute is absent.
    let Some(shape_v) = args.first() else {
        return Eval::Unsupported;
    };
    let dims = shape_v.as_ints();
    if dims.iter().any(|&d| d < 0) {
        return Eval::Unsupported;
    }
    let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    let n: usize = shape.iter().product();
    if n > config.max_elements {
        return Eval::TooLarge;
    }
    let data = match dtype {
        DtypeRepr::F32 | DtypeRepr::F64 => {
            Data::Float(vec![fill.map_or(0.0, |f| f.float_at(0)); n])
        }
        DtypeRepr::Bool => Data::Bool(vec![fill.is_some_and(|f| f.bool_at(0)); n]),
        DtypeRepr::F16 | DtypeRepr::BF16 => return Eval::Unsupported,
        _ => Data::Int(vec![fill.map_or(0, |f| f.int_at(0)); n]),
    };
    Eval::Ok(Value { dtype, shape, data })
}

fn eval_range(args: &[&Value], config: &MaterializeConfig) -> Eval {
    if args.len() != 3 || args.iter().any(|a| a.len() != 1) {
        return Eval::Unsupported;
    }
    let (start, limit, delta) = (args[0], args[1], args[2]);
    let dtype = start.dtype;
    if start.is_float() || limit.is_float() || delta.is_float() {
        let (s, l, d) = (start.float_at(0), limit.float_at(0), delta.float_at(0));
        if d == 0.0 {
            return Eval::Unsupported;
        }
        let n = ((l - s) / d).ceil().max(0.0);
        if !n.is_finite() || n as usize > config.max_elements {
            return Eval::TooLarge;
        }
        let n = n as usize;
        let vals: Vec<f64> = (0..n).map(|i| s + i as f64 * d).collect();
        return Eval::Ok(Value {
            dtype,
            shape: vec![n],
            data: Data::Float(vals),
        });
    }
    let (s, l, d) = (start.int_at(0), limit.int_at(0), delta.int_at(0));
    if d == 0 {
        return Eval::Unsupported;
    }
    // Ceiling division of the span by the step, clamped at zero.
    let span = l - s;
    let n = if (span > 0) == (d > 0) {
        ((span + d - d.signum()) / d).max(0)
    } else {
        0
    };
    if n as usize > config.max_elements {
        return Eval::TooLarge;
    }
    let vals: Vec<i64> = (0..n).map(|i| s + i * d).collect();
    Eval::Ok(Value {
        dtype,
        shape: vec![n as usize],
        data: Data::Int(vals),
    })
}

fn eval_reshape(args: &[&Value]) -> Eval {
    if args.len() != 2 {
        return Eval::Unsupported;
    }
    let (data, target) = (args[0], args[1]);
    let dims = target.as_ints();
    let n = data.len();

    let mut out: Vec<usize> = Vec::with_capacity(dims.len());
    let mut wildcard = None;
    for (ax, &d) in dims.iter().enumerate() {
        match d {
            // `0` means "keep the input's dim at this axis".
            0 => match data.shape.get(ax) {
                Some(&keep) => out.push(keep),
                None => return Eval::Unsupported,
            },
            -1 => {
                if wildcard.is_some() {
                    return Eval::Unsupported;
                }
                wildcard = Some(ax);
                out.push(1);
            }
            d if d > 0 => out.push(d as usize),
            _ => return Eval::Unsupported,
        }
    }
    if let Some(ax) = wildcard {
        let known: usize = out.iter().product();
        if known == 0 || n % known != 0 {
            return Eval::Unsupported;
        }
        out[ax] = n / known;
    }
    if out.iter().product::<usize>() != n {
        return Eval::Unsupported;
    }
    Eval::Ok(data.with_shape(out))
}

fn eval_squeeze(args: &[&Value], axes: &[i64]) -> Eval {
    let Some(v) = args.first() else {
        return Eval::Unsupported;
    };
    let rank = v.rank();
    let mut drop = vec![false; rank];
    if axes.is_empty() {
        for (ax, &d) in v.shape.iter().enumerate() {
            drop[ax] = d == 1;
        }
    } else {
        for &a in axes {
            match norm_axis(a, rank) {
                Some(ax) if ax < rank && v.shape[ax] == 1 => drop[ax] = true,
                _ => return Eval::Unsupported,
            }
        }
    }
    let out: Vec<usize> = v
        .shape
        .iter()
        .enumerate()
        .filter(|(ax, _)| !drop[*ax])
        .map(|(_, &d)| d)
        .collect();
    Eval::Ok(v.with_shape(out))
}

fn eval_unsqueeze(args: &[&Value], axes: &[i64]) -> Eval {
    let Some(v) = args.first() else {
        return Eval::Unsupported;
    };
    let out_rank = v.rank() + axes.len();
    let mut insert = vec![false; out_rank];
    for &a in axes {
        match norm_axis(a, out_rank) {
            Some(ax) if ax < out_rank && !insert[ax] => insert[ax] = true,
            _ => return Eval::Unsupported,
        }
    }
    let mut out = Vec::with_capacity(out_rank);
    let mut src = v.shape.iter();
    for &inserted in &insert {
        if inserted {
            out.push(1);
        } else {
            match src.next() {
                Some(&d) => out.push(d),
                None => return Eval::Unsupported,
            }
        }
    }
    Eval::Ok(v.with_shape(out))
}

fn eval_concat(args: &[&Value], axis: i64, config: &MaterializeConfig) -> Eval {
    if args.is_empty() {
        return Eval::Unsupported;
    }
    // Exporters build shape vectors by concatenating scalars, mixing rank-0 and
    // rank-1 operands in one node. Runtimes read a rank-0 operand as `[1]`; do the
    // same rather than decline, since this is most of the plumbing we are here for.
    let promoted: Vec<Value> = args
        .iter()
        .map(|a| {
            if a.rank() == 0 {
                a.with_shape(vec![1])
            } else {
                (*a).clone()
            }
        })
        .collect();
    let args: Vec<&Value> = promoted.iter().collect();
    let args = args.as_slice();

    let rank = args[0].rank();
    let Some(ax) = norm_axis(axis, rank).filter(|&ax| ax < rank) else {
        return Eval::Unsupported;
    };
    // Every operand must agree on every axis but the concatenated one.
    let mut out_shape = args[0].shape.clone();
    let mut concat_dim = 0;
    for a in args {
        if a.rank() != rank {
            return Eval::Unsupported;
        }
        for (i, (&d, &d0)) in a.shape.iter().zip(args[0].shape.iter()).enumerate() {
            if i != ax && d != d0 {
                return Eval::Unsupported;
            }
        }
        concat_dim += a.shape[ax];
    }
    out_shape[ax] = concat_dim;
    let total: usize = out_shape.iter().product();
    if total > config.max_elements {
        return Eval::TooLarge;
    }

    // Walk the output and pull each element from whichever operand owns that slot.
    let mut offsets = Vec::with_capacity(args.len());
    let mut acc = 0;
    for a in args {
        offsets.push(acc);
        acc += a.shape[ax];
    }

    let mut ints = Vec::with_capacity(total);
    let mut floats = Vec::with_capacity(total);
    let mut bools = Vec::with_capacity(total);
    let float_out = args[0].is_float();
    let bool_out = matches!(args[0].data, Data::Bool(_));

    for flat in 0..total {
        let mut c = coords_of(flat, &out_shape);
        let which = offsets
            .iter()
            .rposition(|&o| o <= c[ax])
            .unwrap_or(0);
        c[ax] -= offsets[which];
        let src = args[which];
        let off: usize = c
            .iter()
            .zip(contiguous_strides(&src.shape))
            .map(|(&ci, s)| ci * s)
            .sum();
        if bool_out {
            bools.push(src.bool_at(off));
        } else if float_out {
            floats.push(src.float_at(off));
        } else {
            ints.push(src.int_at(off));
        }
    }

    let data = if bool_out {
        Data::Bool(bools)
    } else if float_out {
        Data::Float(floats)
    } else {
        Data::Int(ints)
    };
    Eval::Ok(Value {
        dtype: args[0].dtype,
        shape: out_shape,
        data,
    })
}

fn eval_gather(args: &[&Value], axis: i64, config: &MaterializeConfig) -> Eval {
    if args.len() != 2 {
        return Eval::Unsupported;
    }
    let (data, indices) = (args[0], args[1]);
    let rank = data.rank();
    let Some(ax) = norm_axis(axis, rank).filter(|&ax| ax < rank) else {
        return Eval::Unsupported;
    };
    let dim = data.shape[ax] as i64;

    // Output replaces the gathered axis with the indices' own shape.
    let mut out_shape = Vec::with_capacity(rank - 1 + indices.rank());
    out_shape.extend_from_slice(&data.shape[..ax]);
    out_shape.extend_from_slice(&indices.shape);
    out_shape.extend_from_slice(&data.shape[ax + 1..]);
    let total: usize = out_shape.iter().product();
    if total > config.max_elements {
        return Eval::TooLarge;
    }

    let idx: Vec<i64> = indices
        .as_ints()
        .iter()
        .map(|&i| if i < 0 { i + dim } else { i })
        .collect();
    if idx.iter().any(|&i| i < 0 || i >= dim) {
        return Eval::OutOfRange;
    }

    // `data` is contiguous, so the gathered axis sits at a stride of `inner` inside
    // each of the `outer` leading blocks.
    let outer: usize = data.shape[..ax].iter().product();
    let inner: usize = data.shape[ax + 1..].iter().product();
    let block = dim as usize * inner;

    let mut flat = Vec::with_capacity(total);
    for o in 0..outer {
        for &i in &idx {
            for n in 0..inner {
                flat.push(o * block + i as usize * inner + n);
            }
        }
    }
    Eval::Ok(data.gather_flat(&flat, out_shape))
}

fn eval_slice(args: &[&Value]) -> Eval {
    if !(3..=5).contains(&args.len()) {
        return Eval::Unsupported;
    }
    let data = args[0];
    let rank = data.rank();
    let starts = args[1].as_ints();
    let ends = args[2].as_ints();
    let axes: Vec<i64> = match args.get(3) {
        Some(a) => a.as_ints(),
        None => (0..starts.len() as i64).collect(),
    };
    let steps: Vec<i64> = match args.get(4) {
        Some(a) => a.as_ints(),
        None => vec![1; starts.len()],
    };
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return Eval::Unsupported;
    }

    // Per-axis index lists, defaulting to the full extent.
    let mut picks: Vec<Vec<usize>> = data
        .shape
        .iter()
        .map(|&d| (0..d).collect::<Vec<usize>>())
        .collect();

    for k in 0..starts.len() {
        let Some(ax) = norm_axis(axes[k], rank).filter(|&ax| ax < rank) else {
            return Eval::Unsupported;
        };
        let dim = data.shape[ax] as i64;
        let step = steps[k];
        if step == 0 {
            return Eval::Unsupported;
        }
        let mut s = if starts[k] < 0 { starts[k] + dim } else { starts[k] };
        let mut e = if ends[k] < 0 { ends[k] + dim } else { ends[k] };
        let mut out = Vec::new();
        if step > 0 {
            s = s.clamp(0, dim);
            e = e.clamp(0, dim);
            let mut i = s;
            while i < e {
                out.push(i as usize);
                i += step;
            }
        } else {
            // Reverse slicing: ONNX clamps the start one short of the end of the axis.
            s = s.clamp(-1, dim - 1);
            e = e.clamp(-1, dim - 1);
            let mut i = s;
            while i > e {
                if i < 0 {
                    break;
                }
                out.push(i as usize);
                i += step;
            }
        }
        picks[ax] = out;
    }

    let out_shape: Vec<usize> = picks.iter().map(|p| p.len()).collect();
    let total: usize = out_shape.iter().product();
    let strides = contiguous_strides(&data.shape);
    let mut flat = Vec::with_capacity(total);
    for i in 0..total {
        let c = coords_of(i, &out_shape);
        let off: usize = c
            .iter()
            .enumerate()
            .map(|(ax, &ci)| picks[ax][ci] * strides[ax])
            .sum();
        flat.push(off);
    }
    Eval::Ok(data.gather_flat(&flat, out_shape))
}

fn eval_transpose(args: &[&Value], perm: &[usize]) -> Eval {
    let Some(v) = args.first() else {
        return Eval::Unsupported;
    };
    let rank = v.rank();
    if perm.len() != rank || perm.iter().any(|&p| p >= rank) {
        return Eval::Unsupported;
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| v.shape[p]).collect();
    let src_strides = contiguous_strides(&v.shape);
    let total: usize = out_shape.iter().product();
    let mut flat = Vec::with_capacity(total);
    for i in 0..total {
        let c = coords_of(i, &out_shape);
        let off: usize = c
            .iter()
            .zip(perm.iter())
            .map(|(&ci, &p)| ci * src_strides[p])
            .sum();
        flat.push(off);
    }
    Eval::Ok(v.gather_flat(&flat, out_shape))
}

fn eval_expand(args: &[&Value], config: &MaterializeConfig) -> Eval {
    if args.len() != 2 {
        return Eval::Unsupported;
    }
    let (data, target) = (args[0], args[1]);
    let dims = target.as_ints();
    if dims.iter().any(|&d| d < 0) {
        return Eval::Unsupported;
    }
    let want: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    // `Expand` broadcasts both ways: the result can be wider than the target.
    let Some(out_shape) = broadcast_shapes(&data.shape, &want) else {
        return Eval::Unsupported;
    };
    let total: usize = out_shape.iter().product();
    if total > config.max_elements {
        return Eval::TooLarge;
    }
    let mut flat = Vec::with_capacity(total);
    for i in 0..total {
        let c = coords_of(i, &out_shape);
        flat.push(broadcast_offset(&c, &data.shape));
    }
    Eval::Ok(data.gather_flat(&flat, out_shape))
}

fn eval_cast(args: &[&Value], to: DtypeRepr) -> Eval {
    let Some(v) = args.first() else {
        return Eval::Unsupported;
    };
    let n = v.len();
    let data = match to {
        DtypeRepr::Bool => Data::Bool((0..n).map(|i| v.bool_at(i)).collect()),
        DtypeRepr::F32 => Data::Float((0..n).map(|i| v.float_at(i) as f32 as f64).collect()),
        DtypeRepr::F64 => Data::Float((0..n).map(|i| v.float_at(i)).collect()),
        DtypeRepr::F16 | DtypeRepr::BF16 => return Eval::Unsupported,
        // Integer casts truncate toward zero, as ONNX specifies, and then wrap to the
        // target width so the value round-trips through `encode`.
        _ => {
            let ints: Vec<i64> = (0..n)
                .map(|i| {
                    let x = if v.is_float() {
                        v.float_at(i).trunc() as i64
                    } else {
                        v.int_at(i)
                    };
                    narrow(x, to)
                })
                .collect();
            Data::Int(ints)
        }
    };
    Eval::Ok(Value {
        dtype: to,
        shape: v.shape.clone(),
        data,
    })
}

fn narrow(x: i64, to: DtypeRepr) -> i64 {
    match to {
        DtypeRepr::I8 => x as i8 as i64,
        DtypeRepr::U8 => x as u8 as i64,
        DtypeRepr::I16 => x as i16 as i64,
        DtypeRepr::U16 => x as u16 as i64,
        DtypeRepr::I32 => x as i32 as i64,
        DtypeRepr::U32 => x as u32 as i64,
        _ => x,
    }
}

#[derive(Clone, Copy)]
enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod { fmod: bool },
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
}

impl BinKind {
    fn is_comparison(self) -> bool {
        matches!(
            self,
            BinKind::Equal
                | BinKind::Greater
                | BinKind::GreaterOrEqual
                | BinKind::Less
                | BinKind::LessOrEqual
        )
    }
}

fn binary(args: &[&Value], kind: BinKind) -> Eval {
    if args.len() != 2 {
        return Eval::Unsupported;
    }
    let (a, b) = (args[0], args[1]);
    let Some(out_shape) = broadcast_shapes(&a.shape, &b.shape) else {
        return Eval::Unsupported;
    };
    let total: usize = out_shape.iter().product();
    let float_math = a.is_float() || b.is_float();

    let mut ints = Vec::new();
    let mut floats = Vec::new();
    let mut bools = Vec::new();

    for i in 0..total {
        let c = coords_of(i, &out_shape);
        let ia = broadcast_offset(&c, &a.shape);
        let ib = broadcast_offset(&c, &b.shape);

        if kind.is_comparison() {
            let ord = if float_math {
                a.float_at(ia).partial_cmp(&b.float_at(ib))
            } else {
                Some(a.int_at(ia).cmp(&b.int_at(ib)))
            };
            let Some(ord) = ord else {
                return Eval::Unsupported; // NaN comparison; leave the node alone.
            };
            bools.push(match kind {
                BinKind::Equal => ord.is_eq(),
                BinKind::Greater => ord.is_gt(),
                BinKind::GreaterOrEqual => ord.is_ge(),
                BinKind::Less => ord.is_lt(),
                BinKind::LessOrEqual => ord.is_le(),
                _ => unreachable!("non-comparison kind in comparison branch"),
            });
            continue;
        }

        if float_math {
            let (x, y) = (a.float_at(ia), b.float_at(ib));
            floats.push(match kind {
                BinKind::Add => x + y,
                BinKind::Sub => x - y,
                BinKind::Mul => x * y,
                BinKind::Div => x / y,
                BinKind::Pow => x.powf(y),
                // ONNX `Mod` on floats requires fmod=1; its semantics are C's fmod.
                BinKind::Mod { .. } => x % y,
                _ => unreachable!("comparison kind in arithmetic branch"),
            });
        } else {
            let (x, y) = (a.int_at(ia), b.int_at(ib));
            let v = match kind {
                BinKind::Add => x.wrapping_add(y),
                BinKind::Sub => x.wrapping_sub(y),
                BinKind::Mul => x.wrapping_mul(y),
                BinKind::Div | BinKind::Mod { .. } if y == 0 => return Eval::Unsupported,
                // Integer division truncates toward zero in ONNX, as it does in Rust.
                BinKind::Div => x / y,
                BinKind::Pow => {
                    let Ok(e) = u32::try_from(y) else {
                        return Eval::Unsupported;
                    };
                    x.wrapping_pow(e)
                }
                // With fmod=0 the sign follows the divisor (Python); with 1 it follows
                // the dividend (C).
                BinKind::Mod { fmod } => {
                    if fmod {
                        x % y
                    } else {
                        x.rem_euclid(y.abs()) * if y < 0 { -1 } else { 1 }
                    }
                }
                _ => unreachable!("comparison kind in arithmetic branch"),
            };
            ints.push(narrow(v, a.dtype));
        }
    }

    let (dtype, data) = if kind.is_comparison() {
        (DtypeRepr::Bool, Data::Bool(bools))
    } else if float_math {
        (if a.is_float() { a.dtype } else { b.dtype }, Data::Float(floats))
    } else {
        (a.dtype, Data::Int(ints))
    };
    Eval::Ok(Value {
        dtype,
        shape: out_shape,
        data,
    })
}

#[derive(Clone, Copy)]
enum UnKind {
    Neg,
    Abs,
    Sqrt,
    Floor,
    Ceil,
    Not,
}

fn unary(args: &[&Value], kind: UnKind) -> Eval {
    let Some(v) = one(args) else {
        return Eval::Unsupported;
    };
    let n = v.len();
    if matches!(kind, UnKind::Not) {
        return Eval::Ok(Value {
            dtype: DtypeRepr::Bool,
            shape: v.shape.clone(),
            data: Data::Bool((0..n).map(|i| !v.bool_at(i)).collect()),
        });
    }
    // `Sqrt` is float-only in ONNX; the rest keep the operand's lane so integer
    // shape arithmetic stays exact.
    if v.is_float() || matches!(kind, UnKind::Sqrt) {
        let vals: Vec<f64> = (0..n)
            .map(|i| {
                let x = v.float_at(i);
                match kind {
                    UnKind::Neg => -x,
                    UnKind::Abs => x.abs(),
                    UnKind::Sqrt => x.sqrt(),
                    UnKind::Floor => x.floor(),
                    UnKind::Ceil => x.ceil(),
                    UnKind::Not => unreachable!("Not handled above"),
                }
            })
            .collect();
        if vals.iter().any(|x| x.is_nan()) {
            // A NaN here means a negative `Sqrt`; leave the node for the runtime.
            return Eval::Unsupported;
        }
        let dtype = if v.is_float() { v.dtype } else { DtypeRepr::F32 };
        return Eval::Ok(Value {
            dtype,
            shape: v.shape.clone(),
            data: Data::Float(vals),
        });
    }
    let vals: Vec<i64> = (0..n)
        .map(|i| {
            let x = v.int_at(i);
            match kind {
                UnKind::Neg => x.wrapping_neg(),
                UnKind::Abs => x.wrapping_abs(),
                // Already integral.
                UnKind::Floor | UnKind::Ceil => x,
                UnKind::Sqrt | UnKind::Not => unreachable!("handled above"),
            }
        })
        .map(|x| narrow(x, v.dtype))
        .collect();
    Eval::Ok(Value {
        dtype: v.dtype,
        shape: v.shape.clone(),
        data: Data::Int(vals),
    })
}

fn eval_where(args: &[&Value]) -> Eval {
    if args.len() != 3 {
        return Eval::Unsupported;
    }
    let (cond, x, y) = (args[0], args[1], args[2]);
    let Some(s1) = broadcast_shapes(&cond.shape, &x.shape) else {
        return Eval::Unsupported;
    };
    let Some(out_shape) = broadcast_shapes(&s1, &y.shape) else {
        return Eval::Unsupported;
    };
    let total: usize = out_shape.iter().product();
    let float_out = x.is_float() || y.is_float();
    let bool_out = matches!(x.data, Data::Bool(_)) && matches!(y.data, Data::Bool(_));

    let mut ints = Vec::new();
    let mut floats = Vec::new();
    let mut bools = Vec::new();
    for i in 0..total {
        let c = coords_of(i, &out_shape);
        let take_x = cond.bool_at(broadcast_offset(&c, &cond.shape));
        let (src, off) = if take_x {
            (x, broadcast_offset(&c, &x.shape))
        } else {
            (y, broadcast_offset(&c, &y.shape))
        };
        if bool_out {
            bools.push(src.bool_at(off));
        } else if float_out {
            floats.push(src.float_at(off));
        } else {
            ints.push(src.int_at(off));
        }
    }
    let data = if bool_out {
        Data::Bool(bools)
    } else if float_out {
        Data::Float(floats)
    } else {
        Data::Int(ints)
    };
    Eval::Ok(Value {
        dtype: x.dtype,
        shape: out_shape,
        data,
    })
}

/// A short, stable name for an op, for the unsupported-op tally.
fn op_label(op: &Op) -> String {
    let dbg = format!("{op:?}");
    // `Op`'s Debug prints either `Name` or `Name { .. }`; keep just the name.
    dbg.split([' ', '('])
        .next()
        .unwrap_or("Unknown")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_td(shape: &[usize], vals: &[i64]) -> TensorData {
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        TensorData {
            dtype: DtypeRepr::I64,
            shape: shape.iter().map(|&d| Some(d)).collect(),
            bytes,
        }
    }

    fn ints(v: &Value) -> Vec<i64> {
        v.as_ints()
    }

    fn val(shape: &[usize], vals: &[i64]) -> Value {
        Value {
            dtype: DtypeRepr::I64,
            shape: shape.to_vec(),
            data: Data::Int(vals.to_vec()),
        }
    }

    #[test]
    fn shape_of_folds_a_static_shape() {
        let shapes = [vec![Some(1), Some(3), Some(480), Some(640)]];
        let refs: Vec<&Shape> = shapes.iter().collect();
        let Eval::Ok(v) = eval_shape_of(0, i64::MAX, &refs) else {
            panic!("expected a folded shape");
        };
        assert_eq!(ints(&v), vec![1, 3, 480, 640]);
        assert_eq!(v.shape, vec![4]);
    }

    #[test]
    fn shape_of_declines_a_dynamic_dim() {
        let shapes = [vec![None, Some(3)]];
        let refs: Vec<&Shape> = shapes.iter().collect();
        assert!(matches!(
            eval_shape_of(0, i64::MAX, &refs),
            Eval::Unsupported
        ));
    }

    /// An empty recorded shape is ambiguous — a real rank-0 tensor, or one the
    /// importer never inferred — so folding it would risk an empty shape vector.
    #[test]
    fn shape_of_refuses_an_unknown_rank() {
        let shapes = [Shape::new()];
        let refs: Vec<&Shape> = shapes.iter().collect();
        assert!(matches!(
            eval_shape_of(0, i64::MAX, &refs),
            Eval::UnknownRank
        ));
    }

    /// Exporters concatenate scalars to build a shape vector; rank-0 reads as `[1]`.
    #[test]
    fn concat_promotes_scalar_operands() {
        let a = val(&[], &[1]);
        let b = val(&[], &[64]);
        let c = val(&[1], &[-1]);
        let Eval::Ok(v) = eval_concat(&[&a, &b, &c], 0, &MaterializeConfig::default()) else {
            panic!("expected scalars to concatenate");
        };
        assert_eq!(ints(&v), vec![1, 64, -1]);
        assert_eq!(v.shape, vec![3]);
    }

    #[test]
    fn gather_reports_an_out_of_range_index() {
        let data = val(&[4], &[1, 3, 480, 640]);
        let idx = val(&[], &[4]);
        assert!(matches!(
            eval_gather(&[&data, &idx], 0, &MaterializeConfig::default()),
            Eval::OutOfRange
        ));
    }

    #[test]
    fn gather_picks_one_dim_out_of_a_shape_vector() {
        let data = val(&[4], &[1, 3, 480, 640]);
        let idx = val(&[], &[2]);
        let Eval::Ok(v) = eval_gather(&[&data, &idx], 0, &MaterializeConfig::default()) else {
            panic!("expected a folded gather");
        };
        assert_eq!(ints(&v), vec![480]);
        assert!(v.shape.is_empty(), "a scalar index yields a scalar");
    }

    #[test]
    fn concat_builds_a_reshape_target() {
        let a = val(&[1], &[1]);
        let b = val(&[1], &[-1]);
        let c = val(&[2], &[480, 640]);
        let Eval::Ok(v) = eval_concat(&[&a, &b, &c], 0, &MaterializeConfig::default()) else {
            panic!("expected a folded concat");
        };
        assert_eq!(ints(&v), vec![1, -1, 480, 640]);
    }

    #[test]
    fn binary_ops_broadcast_and_divide_toward_zero() {
        let a = val(&[3], &[7, -7, 9]);
        let b = val(&[], &[2]);
        let Eval::Ok(v) = binary(&[&a, &b], BinKind::Div) else {
            panic!("expected a folded div");
        };
        assert_eq!(ints(&v), vec![3, -3, 4]);
        assert_eq!(v.shape, vec![3]);
    }

    #[test]
    fn slice_handles_negative_and_stepped_ranges() {
        let data = val(&[6], &[0, 1, 2, 3, 4, 5]);
        let starts = val(&[1], &[1]);
        let ends = val(&[1], &[-1]);
        let axes = val(&[1], &[0]);
        let steps = val(&[1], &[2]);
        let Eval::Ok(v) = eval_slice(&[&data, &starts, &ends, &axes, &steps]) else {
            panic!("expected a folded slice");
        };
        assert_eq!(ints(&v), vec![1, 3]);
    }

    #[test]
    fn transpose_moves_elements() {
        let data = val(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let Eval::Ok(v) = eval_transpose(&[&data], &[1, 0]) else {
            panic!("expected a folded transpose");
        };
        assert_eq!(v.shape, vec![3, 2]);
        assert_eq!(ints(&v), vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn reshape_resolves_a_wildcard() {
        let data = val(&[6], &[0, 1, 2, 3, 4, 5]);
        let target = val(&[2], &[3, -1]);
        let Eval::Ok(v) = eval_reshape(&[&data, &target]) else {
            panic!("expected a folded reshape");
        };
        assert_eq!(v.shape, vec![3, 2]);
    }

    #[test]
    fn range_counts_up_and_down() {
        let cfg = MaterializeConfig::default();
        let (s, l, d) = (val(&[], &[0]), val(&[], &[5]), val(&[], &[2]));
        let Eval::Ok(v) = eval_range(&[&s, &l, &d], &cfg) else {
            panic!("expected a folded range");
        };
        assert_eq!(ints(&v), vec![0, 2, 4]);

        let (s, l, d) = (val(&[], &[5]), val(&[], &[0]), val(&[], &[-2]));
        let Eval::Ok(v) = eval_range(&[&s, &l, &d], &cfg) else {
            panic!("expected a folded descending range");
        };
        assert_eq!(ints(&v), vec![5, 3, 1]);
    }

    #[test]
    fn expand_stretches_size_one_axes() {
        let data = val(&[1, 3], &[7, 8, 9]);
        let target = val(&[2], &[2, 3]);
        let Eval::Ok(v) = eval_expand(&[&data, &target], &MaterializeConfig::default()) else {
            panic!("expected a folded expand");
        };
        assert_eq!(v.shape, vec![2, 3]);
        assert_eq!(ints(&v), vec![7, 8, 9, 7, 8, 9]);
    }

    #[test]
    fn unary_sqrt_promotes_integers_to_float() {
        let v = val(&[2], &[4, 9]);
        let Eval::Ok(r) = unary(&[&v], UnKind::Sqrt) else {
            panic!("expected a folded sqrt");
        };
        assert_eq!(r.dtype, DtypeRepr::F32);
        assert_eq!((0..2).map(|i| r.float_at(i)).collect::<Vec<_>>(), vec![2.0, 3.0]);
    }

    #[test]
    fn unary_neg_stays_integral() {
        let v = val(&[2], &[4, -9]);
        let Eval::Ok(r) = unary(&[&v], UnKind::Neg) else {
            panic!("expected a folded neg");
        };
        assert_eq!(ints(&r), vec![-4, 9]);
        assert_eq!(r.dtype, DtypeRepr::I64);
    }

    /// `ConstantOfShape` emits one element per slot in its shape operand, filled
    /// with its `value` attribute. Standing a scalar fill in for the whole tensor
    /// is what made exporter-emitted pad lists come out one element short.
    #[test]
    fn constant_of_shape_expands_to_its_shape_operand() {
        let shape = val(&[1], &[2]);
        let fill = val(&[1], &[0]);
        let Eval::Ok(v) = eval_constant_of_shape(
            DtypeRepr::I64,
            &[&shape],
            Some(&fill),
            &MaterializeConfig::default(),
        ) else {
            panic!("expected a folded ConstantOfShape");
        };
        assert_eq!(v.shape, vec![2]);
        assert_eq!(ints(&v), vec![0, 0]);
    }

    #[test]
    fn constant_of_shape_honours_a_non_zero_fill() {
        let shape = val(&[1], &[3]);
        let fill = val(&[1], &[7]);
        let Eval::Ok(v) = eval_constant_of_shape(
            DtypeRepr::I64,
            &[&shape],
            Some(&fill),
            &MaterializeConfig::default(),
        ) else {
            panic!("expected a folded ConstantOfShape");
        };
        assert_eq!(ints(&v), vec![7, 7, 7]);
    }

    /// The exporter's `F.pad` pattern: a literal prefix concatenated with a
    /// `ConstantOfShape` tail, reshaped into (begin, end) pairs. The whole chain
    /// must fold, and the pad list must come out even.
    #[test]
    fn folds_an_exporter_pad_list() {
        let mut g = Graph::new();
        let shape_c = g.add_node(
            Op::Constant {
                dtype: DtypeRepr::I64,
                shape: vec![Some(1)],
            },
            vec![],
            DtypeRepr::I64,
            vec![Some(1)],
        );
        let cos = g.add_node(
            Op::ConstantOfShape {
                dtype: DtypeRepr::I64,
            },
            vec![shape_c],
            DtypeRepr::I64,
            vec![Some(2)],
        );
        let prefix = g.add_node(
            Op::Constant {
                dtype: DtypeRepr::I64,
                shape: vec![Some(6)],
            },
            vec![],
            DtypeRepr::I64,
            vec![Some(6)],
        );
        let cat = g.add_node(
            Op::Concat { axis: 0 },
            vec![prefix, cos],
            DtypeRepr::I64,
            vec![Some(8)],
        );
        let tgt = g.add_node(
            Op::Constant {
                dtype: DtypeRepr::I64,
                shape: vec![Some(2)],
            },
            vec![],
            DtypeRepr::I64,
            vec![Some(2)],
        );
        let rs = g.add_node(Op::Reshape, vec![cat, tgt], DtypeRepr::I64, vec![Some(4), Some(2)]);

        let mut tensors = BTreeMap::new();
        tensors.insert(shape_c, int_td(&[1], &[2]));
        tensors.insert(cos, int_td(&[1], &[0])); // the fill value
        tensors.insert(prefix, int_td(&[6], &[0, 0, 0, 0, 1, 1]));
        tensors.insert(tgt, int_td(&[2], &[-1, 2]));

        let (folded, report) =
            materialize_constants(&mut g, &tensors, &MaterializeConfig::default());
        assert!(
            report.unsupported.is_empty(),
            "the pad chain should fold whole: {:?}",
            report.unsupported_ranked()
        );
        let v = Value::decode(&folded[&rs]).expect("reshape result decodes");
        assert_eq!(v.shape, vec![4, 2], "an 8-element pad list, not 7");
        assert_eq!(v.as_ints(), vec![0, 0, 0, 0, 1, 1, 0, 0]);
    }

    #[test]
    fn values_round_trip_through_encode() {
        let v = val(&[2, 2], &[1, -2, 3, -4]);
        let decoded = Value::decode(&v.encode()).expect("i64 decodes");
        assert_eq!(decoded, v);
    }

    /// The pass folds a whole `Shape -> Gather -> Unsqueeze -> Concat` chain, the
    /// shape of exporter-emitted plumbing, and leaves the weight input alone.
    #[test]
    fn folds_an_exporter_shape_chain_end_to_end() {
        let mut g = Graph::new();
        let x = g.add_node(
            Op::Input,
            vec![],
            DtypeRepr::F32,
            vec![Some(1), Some(64), Some(480), Some(640)],
        );
        let shape = g.add_node(
            Op::ShapeOf {
                start: 0,
                end: i64::MAX,
            },
            vec![x],
            DtypeRepr::I64,
            vec![Some(4)],
        );
        let idx = g.add_node(Op::Constant {
            dtype: DtypeRepr::I64,
            shape: vec![],
        }, vec![], DtypeRepr::I64, vec![]);
        let dim = g.add_node(Op::Gather { axis: 0 }, vec![shape, idx], DtypeRepr::I64, vec![]);
        let unsq = g.add_node(
            Op::Unsqueeze { axes: vec![0] },
            vec![dim],
            DtypeRepr::I64,
            vec![Some(1)],
        );
        let tail = g.add_node(Op::Constant {
            dtype: DtypeRepr::I64,
            shape: vec![Some(1)],
        }, vec![], DtypeRepr::I64, vec![Some(1)]);
        let cat = g.add_node(
            Op::Concat { axis: 0 },
            vec![unsq, tail],
            DtypeRepr::I64,
            vec![Some(2)],
        );

        let mut tensors = BTreeMap::new();
        tensors.insert(idx, int_td(&[], &[2])); // gather dim 2 (= 480)
        tensors.insert(tail, int_td(&[1], &[-1]));

        let (folded, report) = materialize_constants(&mut g, &tensors, &MaterializeConfig::default());

        assert_eq!(report.folded, 4, "Shape, Gather, Unsqueeze and Concat fold");
        assert!(
            report.unsupported.is_empty(),
            "unexpected unsupported ops: {:?}",
            report.unsupported_ranked()
        );
        // The chain collapses to the reshape target [480, -1].
        let cat_v = Value::decode(&folded[&cat]).expect("concat result decodes");
        assert_eq!(cat_v.as_ints(), vec![480, -1]);
        assert!(matches!(g.nodes[cat].op, Op::Constant { .. }));
        assert!(g.nodes[cat].inputs.is_empty());
        // The weight stays an input, so it can still be bound by name.
        assert!(matches!(g.nodes[x].op, Op::Input));
    }

    #[test]
    fn large_tensors_are_never_decoded() {
        let mut g = Graph::new();
        let w = g.add_node(Op::Input, vec![], DtypeRepr::F32, vec![Some(1024)]);
        let id = g.add_node(Op::Identity, vec![w], DtypeRepr::F32, vec![Some(1024)]);

        let mut tensors = BTreeMap::new();
        tensors.insert(
            w,
            TensorData {
                dtype: DtypeRepr::F32,
                shape: vec![Some(1024)],
                bytes: vec![0u8; 4096],
            },
        );

        let cfg = MaterializeConfig { max_elements: 16 };
        let (folded, report) = materialize_constants(&mut g, &tensors, &cfg);
        assert!(folded.is_empty(), "nothing above the ceiling is folded");
        assert_eq!(report.folded, 0);
        // The `Identity` never became evaluable, because its operand was never decoded.
        assert!(matches!(g.nodes[id].op, Op::Identity));
    }
}
