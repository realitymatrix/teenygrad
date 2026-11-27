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

use teeny_core::device::context::{Context, DeviceInfo};

use crate::device::X86Device;
use crate::errors::Result;

/// A x86-64 "device"'s identifying metadata. Synthetic: unlike CUDA there is no hardware to
/// query -- there is exactly one "device", the local machine (native x86-64) or the
/// `qemu-riscv64` process this code is running under/targeting.
#[derive(Debug, Clone)]
pub struct X86DeviceInfo {
    id: i32,
    name: String,
}

impl Default for X86DeviceInfo {
    fn default() -> Self {
        Self {
            id: 0,
            name: "x86_64 host".to_string(),
        }
    }
}

impl DeviceInfo for X86DeviceInfo {
    type Id = i32;

    fn id(&self) -> Self::Id {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// The x86-64 [`Context`]: entry point for listing/opening devices.
pub struct X86<'a> {
    _unused: PhantomData<&'a ()>,
}

impl<'a> X86<'a> {
    /// Creates the x86-64 context. Infallible -- unlike CUDA there is no driver to initialize.
    pub fn try_new() -> Result<Self> {
        Ok(Self {
            _unused: PhantomData,
        })
    }
}

impl<'a> Context<'a> for X86<'a> {
    type Device = X86Device<'a>;
    type DeviceInfo = X86DeviceInfo;

    /// Always returns the single synthetic device -- see [`X86DeviceInfo`]'s doc comment.
    fn list_devices(&self) -> Result<Vec<Self::DeviceInfo>> {
        Ok(vec![X86DeviceInfo::default()])
    }

    fn device(&self, id: &<Self::DeviceInfo as DeviceInfo>::Id) -> Result<Self::Device> {
        Ok(X86Device::new(X86DeviceInfo {
            id: *id,
            ..X86DeviceInfo::default()
        }))
    }
}
