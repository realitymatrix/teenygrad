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

//! Assigns a kernel's arguments to x86-64 System V calling-convention slots, and launches
//! kernels in-process.
//!
//! As on RISC-V, a kernel's argument list is only known at run time, so [`Frame::new`] lays the
//! arguments out as the SysV ABI would for a call to the kernel's real signature and the launcher
//! calls every entry point through one fixed signature: six integer registers
//! (`rdi rsi rdx rcx r8 r9`), eight SSE registers (`xmm0`-`xmm7`), then [`STACK_SLOTS`] eightbyte
//! stack slots. The callee reads only what its real signature uses; the caller owns the outgoing
//! argument area, so the extra slots are harmless.
//!
//! SysV differs from RISC-V `lp64d` in two ways that matter here: a float that finds no free SSE
//! register goes to the stack (never to an integer register), and an `f32` sits in the low 32
//! bits of its register with no NaN-boxing. Integer and float arguments that overflow share the
//! stack in argument order.

use teeny_core::device::program::ArgVisitor;

/// Integer argument registers: `rdi rsi rdx rcx r8 r9`.
pub(crate) const INT_REGS: usize = 6;
/// SSE argument registers: `xmm0`-`xmm7`.
pub(crate) const SSE_REGS: usize = 8;
/// Stack slots the fixed call signature passes.
pub(crate) const STACK_SLOTS: usize = 32;

/// One kernel argument, in the SysV classes that decide where it goes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Slot {
    /// INTEGER class (pointer, bool, integer), widened to 64 bits.
    Int(u64),
    /// An `f32` (SSE class).
    F32(f32),
    /// An `f64` (SSE class).
    F64(f64),
}

/// A kernel call's arguments, assigned to registers and stack slots.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Frame {
    pub(crate) int_regs: [u64; INT_REGS],
    /// Raw 64-bit contents; an `f32` occupies the low 32 bits.
    pub(crate) sse_regs: [u64; SSE_REGS],
    pub(crate) stack: [u64; STACK_SLOTS],
}

impl Frame {
    /// Lays out `args` followed by TritonCPU's program-id/grid arguments (`pid_x/y/z` as `i32`,
    /// then `grid_x/y/z`), with the program-id slots zeroed for [`Self::set_program_id`].
    ///
    /// Returns `None` if the arguments need more than [`STACK_SLOTS`] stack slots.
    pub(crate) fn new(args: &[Slot], grid: [u32; 3]) -> Option<(Frame, ProgramIdSlots)> {
        let mut frame = Frame {
            int_regs: [0; INT_REGS],
            sse_regs: [0; SSE_REGS],
            stack: [0; STACK_SLOTS],
        };
        let (mut next_int, mut next_sse, mut next_stack) = (0, 0, 0);

        let mut place = |slot: Slot| -> Option<Location> {
            let (bits, is_float) = match slot {
                Slot::Int(v) => (v, false),
                Slot::F32(v) => (u64::from(v.to_bits()), true),
                Slot::F64(v) => (v.to_bits(), true),
            };
            if is_float && next_sse < SSE_REGS {
                frame.sse_regs[next_sse] = bits;
                next_sse += 1;
                return Some(Location::SseReg(next_sse - 1));
            }
            if !is_float && next_int < INT_REGS {
                frame.int_regs[next_int] = bits;
                next_int += 1;
                return Some(Location::IntReg(next_int - 1));
            }
            if next_stack < STACK_SLOTS {
                frame.stack[next_stack] = bits;
                next_stack += 1;
                return Some(Location::Stack(next_stack - 1));
            }
            None
        };

        for &arg in args {
            place(arg)?;
        }
        let pid = [
            place(Slot::Int(0))?,
            place(Slot::Int(0))?,
            place(Slot::Int(0))?,
        ];
        for n in grid {
            place(int_u32(n))?;
        }
        Some((frame, ProgramIdSlots(pid)))
    }

    /// Sets the program id this frame calls the kernel with.
    pub(crate) fn set_program_id(&mut self, slots: &ProgramIdSlots, pid: [u32; 3]) {
        for (location, id) in slots.0.iter().zip(pid) {
            let value = int_i32(id as i32);
            match *location {
                Location::IntReg(i) => self.int_regs[i] = value,
                Location::Stack(i) => self.stack[i] = value,
                Location::SseReg(_) => unreachable!("program ids are integer-class"),
            }
        }
    }
}

/// Where [`Frame::new`] placed the three program-id arguments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ProgramIdSlots([Location; 3]);

#[derive(Debug, Clone, Copy, PartialEq)]
enum Location {
    IntReg(usize),
    SseReg(usize),
    Stack(usize),
}

/// SysV leaves the upper bits of a sub-64-bit integer argument undefined; extending per
/// signedness (what clang and GCC do) is always valid for the callee.
fn int_i32(v: i32) -> u64 {
    i64::from(v) as u64
}

fn int_u32(v: u32) -> Slot {
    Slot::Int(u64::from(v))
}

/// Collects a kernel's arguments as [`Slot`]s.
#[derive(Default)]
pub(crate) struct SlotCollector(pub(crate) Vec<Slot>);

impl SlotCollector {
    /// The SysV slots for `args`.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn collect<A: teeny_core::device::program::KernelArgs>(args: &A) -> Vec<Slot> {
        let mut collector = Self::default();
        args.visit_args(&mut collector);
        collector.0
    }
}

impl ArgVisitor for SlotCollector {
    fn visit_ptr(&mut self, ptr: *mut core::ffi::c_void) {
        self.0.push(Slot::Int(ptr as u64));
    }
    fn visit_bool(&mut self, val: bool) {
        self.0.push(Slot::Int(u64::from(val)));
    }
    fn visit_i8(&mut self, val: i8) {
        self.0.push(Slot::Int(int_i32(val.into())));
    }
    fn visit_i16(&mut self, val: i16) {
        self.0.push(Slot::Int(int_i32(val.into())));
    }
    fn visit_i32(&mut self, val: i32) {
        self.0.push(Slot::Int(int_i32(val)));
    }
    fn visit_i64(&mut self, val: i64) {
        self.0.push(Slot::Int(val as u64));
    }
    fn visit_u8(&mut self, val: u8) {
        self.0.push(Slot::Int(val.into()));
    }
    fn visit_u16(&mut self, val: u16) {
        self.0.push(Slot::Int(val.into()));
    }
    fn visit_u32(&mut self, val: u32) {
        self.0.push(int_u32(val));
    }
    fn visit_u64(&mut self, val: u64) {
        self.0.push(Slot::Int(val));
    }
    fn visit_f32(&mut self, val: f32) {
        self.0.push(Slot::F32(val));
    }
    fn visit_f64(&mut self, val: f64) {
        self.0.push(Slot::F64(val));
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use launcher::{Libraries, launch};

/// Every kernel entry point, called with all six integer registers, all eight SSE registers and
/// [`STACK_SLOTS`] stack slots filled from a [`Frame`].
#[cfg(target_arch = "x86_64")]
#[rustfmt::skip]
type Entry = unsafe extern "C" fn(
    u64, u64, u64, u64, u64, u64,
    f64, f64, f64, f64, f64, f64, f64, f64,
    u64, u64, u64, u64, u64, u64, u64, u64,
    u64, u64, u64, u64, u64, u64, u64, u64,
    u64, u64, u64, u64, u64, u64, u64, u64,
    u64, u64, u64, u64, u64, u64, u64, u64,
);
#[cfg(target_arch = "x86_64")]
const _: () = assert!(STACK_SLOTS == 32);

/// Calls `frame`'s entry point. SSE registers are passed as `f64`s carrying the frame's raw
/// bits; an `f32` argument is read from the low half, which `from_bits` preserves.
///
/// # Safety
/// `entry` must be a function whose arguments `frame` was laid out for.
#[cfg(target_arch = "x86_64")]
#[rustfmt::skip]
unsafe fn call(entry: Entry, frame: &Frame) {
    let a = &frame.int_regs;
    let f = frame.sse_regs.map(f64::from_bits);
    let s = &frame.stack;
    unsafe {
        entry(
            a[0], a[1], a[2], a[3], a[4], a[5],
            f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
            s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15],
            s[16], s[17], s[18], s[19], s[20], s[21], s[22], s[23],
            s[24], s[25], s[26], s[27], s[28], s[29], s[30], s[31],
        )
    }
}

#[cfg(target_arch = "x86_64")]
mod launcher {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};
    use std::time::Duration;

    use anyhow::{Result, anyhow, bail};
    use teeny_core::device::program::KernelArgs;

    use super::{Entry, Frame, STACK_SLOTS, SlotCollector, call};
    use crate::runtime::KernelLibrary;
    use crate::spin_pool::SpinPool;

    /// Kernel libraries this process has loaded, and their resolved entry points.
    #[derive(Default)]
    pub(crate) struct Libraries(Mutex<LoadedLibraries>);

    #[derive(Default)]
    pub(crate) struct LoadedLibraries {
        libraries: HashMap<PathBuf, Arc<KernelLibrary>>,
        entries: HashMap<(PathBuf, String), Entry>,
    }

    impl Libraries {
        fn entry(&self, path: &Path, name: &str) -> Result<Entry> {
            let key = (path.to_path_buf(), name.to_string());
            if let Some(entry) = self
                .0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entries
                .get(&key)
            {
                return Ok(*entry);
            }
            let library = self.get(path)?;
            // Safety: TritonCPU's entry point takes the kernel's arguments then the program id
            // and grid, as the frame lays out; the library is never unloaded.
            let entry: Entry = unsafe { std::mem::transmute(library.symbol(name)?) };
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entries
                .insert(key, entry);
            Ok(entry)
        }

        fn get(&self, path: &Path) -> Result<Arc<KernelLibrary>> {
            let mut loaded = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(library) = loaded.libraries.get(path) {
                return Ok(Arc::clone(library));
            }
            load_libm()?;
            let library = Arc::new(KernelLibrary::load(path)?);
            loaded.libraries.insert(path.to_path_buf(), Arc::clone(&library));
            Ok(library)
        }
    }

    /// Kernels call libm (e.g. `expf`) without listing it as a dependency.
    fn load_libm() -> Result<()> {
        static LIBM: OnceLock<std::result::Result<(), String>> = OnceLock::new();
        LIBM.get_or_init(|| {
            use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
            // Safety: libm's initializers are the system's own.
            let libm = unsafe { Library::open(Some("libm.so.6"), RTLD_NOW | RTLD_GLOBAL) }
                .map_err(|e| format!("dlopen(libm.so.6) failed: {e}"))?;
            std::mem::forget(libm);
            Ok(())
        })
        .clone()
        .map_err(|e| anyhow!(e))
    }

    /// The launch pool: `TEENY_X86_THREADS` threads counting the caller (default: every core
    /// the process may use). Workers spin `TEENY_X86_SPIN_US` microseconds (default 200)
    /// between jobs; see [`SpinPool`].
    fn pool() -> &'static SpinPool {
        static POOL: OnceLock<SpinPool> = OnceLock::new();
        POOL.get_or_init(|| {
            let env = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok());
            let threads = env("TEENY_X86_THREADS")
                .or_else(|| std::thread::available_parallelism().ok().map(Into::into))
                .unwrap_or(1)
                .max(1);
            let spin = Duration::from_micros(env("TEENY_X86_SPIN_US").unwrap_or(200) as u64);
            SpinPool::new(threads, spin, |_| {})
        })
    }

    /// Runs `entry_point` from `library` once for every program id in `grid`, in parallel on
    /// the [`pool`]. Pointers in `args` address host memory the kernel uses directly.
    pub(crate) fn launch<A: KernelArgs>(
        libraries: &Libraries,
        library: &Path,
        entry_point: &str,
        grid: [u32; 3],
        args: &A,
    ) -> Result<()> {
        let slots = SlotCollector::collect(args);
        let Some((frame, pid_slots)) = Frame::new(&slots, grid) else {
            bail!(
                "{entry_point} takes {} arguments, more than the x86 launcher passes on the \
                 stack ({STACK_SLOTS} slots)",
                slots.len()
            );
        };
        let entry = libraries.entry(library, entry_point)?;

        let [gx, gy, gz] = grid.map(u64::from);
        let total = gx * gy * gz;
        let pool = pool();
        let threads = (pool.threads() as u64).min(total).max(1);
        let next = AtomicU64::new(0);
        let run = || {
            let mut frame = frame.clone();
            let chunk = (total / (threads * 8)).max(1);
            loop {
                let start = next.fetch_add(chunk, Ordering::Relaxed);
                if start >= total {
                    break;
                }
                for id in start..(start + chunk).min(total) {
                    let pid = [id % gx, (id / gx) % gy, id / (gx * gy)].map(|n| n as u32);
                    frame.set_program_id(&pid_slots, pid);
                    // Safety: see `entry` above.
                    unsafe { call(entry, &frame) };
                }
            }
        };
        if threads == 1 {
            run();
        } else {
            pool.run(&run);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(args: &[Slot]) -> Frame {
        let (mut frame, pid) = Frame::new(args, [4, 5, 6]).unwrap();
        frame.set_program_id(&pid, [1, 2, 3]);
        frame
    }

    #[test]
    fn integers_fill_six_registers_then_the_stack() {
        let args: Vec<Slot> = (10..14).map(Slot::Int).collect();
        let frame = layout(&args);
        assert_eq!(frame.int_regs, [10, 11, 12, 13, 1, 2]);
        assert_eq!(&frame.stack[..5], &[3, 4, 5, 6, 0]);
    }

    #[test]
    fn floats_use_sse_registers_without_nan_boxing() {
        let frame = layout(&[Slot::Int(7), Slot::F32(1.5), Slot::F64(2.5), Slot::Int(8)]);
        assert_eq!(&frame.int_regs[..2], &[7, 8]);
        assert_eq!(frame.sse_regs[0], u64::from(1.5f32.to_bits()));
        assert_eq!(frame.sse_regs[1], 2.5f64.to_bits());
        assert_eq!(frame.int_regs[2..], [1, 2, 3, 4]);
        assert_eq!(&frame.stack[..2], &[5, 6]);
    }

    #[test]
    fn floats_past_xmm7_go_to_the_stack_not_integer_registers() {
        let mut args: Vec<Slot> = (0..SSE_REGS).map(|i| Slot::F32(i as f32)).collect();
        args.push(Slot::F32(9.0));
        args.push(Slot::Int(42));
        let frame = layout(&args);
        assert_eq!(frame.stack[0], u64::from(9.0f32.to_bits()));
        assert_eq!(frame.int_regs, [42, 1, 2, 3, 4, 5]);
        assert_eq!(frame.stack[1], 6);
    }

    #[test]
    fn too_many_arguments_is_an_error() {
        let args: Vec<Slot> = (0..(INT_REGS + STACK_SLOTS) as u64).map(Slot::Int).collect();
        assert!(Frame::new(&args, [1, 1, 1]).is_none());
    }

    /// Calls a real `extern "C"` function through the fixed signature, with a mix of argument
    /// classes that spills to the stack, to check the frame matches the platform ABI.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fixed_signature_call_matches_the_platform_abi() {
        #[allow(clippy::too_many_arguments)]
        extern "C" fn kernel(
            out: *mut u64,
            b: i32,
            x: f32,
            c: i32,
            d: i32,
            e: i32,
            f: i32,
            g: i32,
            y: f64,
            pid_x: i32,
            pid_y: i32,
            pid_z: i32,
            gx: u32,
            gy: u32,
            gz: u32,
        ) {
            let ints = b + c + d + e + f + g + pid_x * 100 + pid_y * 1000 + pid_z * 10000;
            let v = ints as u64 + (x * 10.0) as u64 + (y * 10.0) as u64 + u64::from(gx + gy + gz);
            unsafe { *out = v };
        }
        let mut out = 0u64;
        let args = [
            Slot::Int(&mut out as *mut u64 as u64),
            Slot::Int(1),
            Slot::F32(0.5),
            Slot::Int(2),
            Slot::Int(3),
            Slot::Int(4),
            Slot::Int(5),
            Slot::Int(6),
            Slot::F64(1.5),
        ];
        let (mut frame, pid) = Frame::new(&args, [7, 8, 9]).unwrap();
        frame.set_program_id(&pid, [1, 2, 3]);
        // Safety: `kernel`'s signature is what `frame` was laid out for.
        unsafe {
            let entry: Entry = std::mem::transmute(kernel as *const ());
            call(entry, &frame);
        }
        // ints 1..=6 = 21, pids 32100, x 5, y 15, grid 24.
        assert_eq!(out, 21 + 32100 + 5 + 15 + 24);
    }
}
