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

/// x86-64 CPU level passed to `teenyc` as `-C target-cpu` for the `x86_64-teeny-kernel` target.
/// Unlike RISC-V's chip identifiers these are real LLVM cpu names, which teenyc's `X86Backend`
/// passes straight to LLVM (`native` is resolved to the host's cpu there).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// `x86-64-v3`: AVX2, FMA, F16C, BMI (Haswell / Zen 1 and later).
    X86_64V3,
    /// `x86-64-v3` plus AVX-VNNI (VEX-encoded `vpdpbusd`): Alder Lake and later client cores,
    /// e.g. Panther Lake (Core Ultra 300).
    X86_64V3Vnni,
    /// `x86-64-v4`: AVX-512 F/BW/DQ/VL.
    X86_64V4,
    /// The host this compiler runs on, as LLVM detects it.
    Native,
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::X86_64V3 => "x86-64-v3",
            Self::X86_64V3Vnni => "x86-64-v3+avxvnni",
            Self::X86_64V4 => "x86-64-v4",
            Self::Native => "native",
        })
    }
}

impl std::str::FromStr for Capability {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "x86-64-v3" => Ok(Self::X86_64V3),
            "x86-64-v3+avxvnni" => Ok(Self::X86_64V3Vnni),
            "x86-64-v4" => Ok(Self::X86_64V4),
            "native" => Ok(Self::Native),
            other => Err(format!(
                "unknown x86 capability '{other}'; expected one of x86-64-v3, \
                 x86-64-v3+avxvnni, x86-64-v4, native"
            )),
        }
    }
}

/// An x86-64 compilation target: a single CPU [`Capability`].
pub struct Target {
    /// The target CPU level.
    pub capability: Capability,
}

impl Target {
    /// Creates a target for the given `capability`.
    pub fn new(capability: Capability) -> Self {
        Self { capability }
    }
}

impl Capability {
    /// The LLVM cpu name for `-C target-cpu`.
    pub fn llvm_cpu(&self) -> &'static str {
        match self {
            Self::X86_64V3 | Self::X86_64V3Vnni => "x86-64-v3",
            Self::X86_64V4 => "x86-64-v4",
            Self::Native => "native",
        }
    }

    /// LLVM target features beyond what the cpu name implies.
    pub fn target_features(&self) -> &'static [&'static str] {
        match self {
            Self::X86_64V3Vnni => &["avxvnni"],
            _ => &[],
        }
    }
}

impl teeny_core::compiler::Target for Target {
    fn target_cpu(&self) -> Option<String> {
        Some(self.capability.llvm_cpu().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::Capability;

    #[test]
    fn display_round_trips_through_from_str() {
        for cap in [
            Capability::X86_64V3,
            Capability::X86_64V3Vnni,
            Capability::X86_64V4,
            Capability::Native,
        ] {
            assert_eq!(cap.to_string().parse::<Capability>().unwrap(), cap);
        }
    }

    #[test]
    fn vnni_is_a_feature_on_v3() {
        assert_eq!(Capability::X86_64V3Vnni.llvm_cpu(), "x86-64-v3");
        assert_eq!(Capability::X86_64V3Vnni.target_features(), &["avxvnni"]);
    }
}
