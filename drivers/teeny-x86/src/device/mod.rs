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

use std::marker::PhantomData;
use std::sync::Arc;

use teeny_core::device::program::Kernel;
use teeny_core::device::{Device, LaunchConfig};
use teeny_core::dtype::Num;

use crate::device::buffer::{BufferRegistry, X86Buffer};
use crate::device::context::X86DeviceInfo;
use crate::device::program::X86Program;
use crate::errors::Result;

/// Device-side memory buffers: host memory, since kernels run in this process.
pub mod buffer;
/// Device/context management.
pub mod context;
/// Compiled kernel programs.
pub mod program;

/// An x86 kernel launch's configuration: the Triton launch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct X86LaunchConfig {
    /// The number of program ids along the x, y and z axes.
    pub grid: [u32; 3],
}

impl X86LaunchConfig {
    /// A launch over `grid` program ids.
    pub fn new(grid: [u32; 3]) -> Self {
        Self { grid }
    }
}

impl Default for X86LaunchConfig {
    /// A single program id.
    fn default() -> Self {
        Self::new([1, 1, 1])
    }
}

impl LaunchConfig for X86LaunchConfig {}

/// The local x86-64 machine as a device: kernels run in-process on its cores.
pub struct X86Device<'a> {
    /// This device's (synthetic) static properties.
    pub info: X86DeviceInfo,
    /// The buffers this device has allocated.
    buffers: Arc<BufferRegistry>,
    /// The kernel libraries launches have loaded.
    #[cfg(target_arch = "x86_64")]
    libraries: crate::native::Libraries,
    _unused: PhantomData<&'a ()>,
}

impl<'a> X86Device<'a> {
    /// Opens the (sole, synthetic) x86 device.
    pub fn new(info: X86DeviceInfo) -> Self {
        Self {
            info,
            buffers: Arc::default(),
            #[cfg(target_arch = "x86_64")]
            libraries: Default::default(),
            _unused: PhantomData,
        }
    }

    /// This device's static properties.
    pub fn info(&self) -> &X86DeviceInfo {
        &self.info
    }
}

impl<'a> Device<'a> for X86Device<'a> {
    type Buffer<N: Num> = X86Buffer<'a, N>;
    type Program<K: Kernel> = X86Program<'a, K>;
    type LaunchConfig = X86LaunchConfig;

    /// Allocates a zero-initialized host buffer -- host memory *is* the device memory here.
    fn buffer<N: Num>(&self, count: usize) -> Result<Self::Buffer<N>> {
        Ok(X86Buffer::with_registry(count, Arc::clone(&self.buffers)))
    }

    /// Runs `program` once for every program id in `cfg.grid`, in this process, spread over
    /// every core (or `TEENY_X86_THREADS` threads).
    ///
    /// Every pointer in `args` must address memory that stays valid for the whole launch.
    #[cfg(target_arch = "x86_64")]
    fn launch<K: Kernel>(
        &self,
        program: &Self::Program<K>,
        cfg: &Self::LaunchConfig,
        args: K::Args<'a>,
    ) -> Result<()> {
        crate::native::launch(
            &self.libraries,
            program.path(),
            program.entry_point(),
            cfg.grid,
            &args,
        )
    }

    /// Always panics: x86 kernels can only run on an x86-64 host.
    #[cfg(not(target_arch = "x86_64"))]
    fn launch<K: Kernel>(
        &self,
        _program: &Self::Program<K>,
        _cfg: &Self::LaunchConfig,
        _args: K::Args<'a>,
    ) -> Result<()> {
        panic!("x86 kernels can only be launched on an x86-64 host")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::buffer::Region;
    use crate::device::context::X86;
    use teeny_core::device::buffer::Buffer;
    use teeny_core::device::context::Context;

    #[test]
    fn buffer_round_trips_to_device_and_to_host() {
        let device = X86Device::new(X86DeviceInfo::default());
        let mut buf = device.buffer::<f32>(4).unwrap();
        let input = [1.0f32, 2.0, 3.0, 4.0];
        buf.to_device(&input).unwrap();
        let mut output = [0.0f32; 4];
        buf.to_host(&mut output).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn device_buffers_are_registered_until_dropped() {
        let device = X86Device::new(X86DeviceInfo::default());
        let buf = device.buffer::<f32>(4).unwrap();
        let addr = buf.as_device_ptr() as usize;
        let region = Region { addr, bytes: 16 };
        assert_eq!(device.buffers.find(addr), Some(region));
        assert_eq!(device.buffers.find(addr + 16), None);
        drop(buf);
        assert_eq!(device.buffers.find(addr), None);
    }

    #[test]
    fn context_lists_exactly_one_synthetic_device() {
        let ctx = X86::try_new().unwrap();
        assert_eq!(ctx.list_devices().unwrap().len(), 1);
    }
}
