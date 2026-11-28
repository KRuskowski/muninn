//! Instruction-set selection and the table of resolved kernels.

use std::sync::OnceLock;

use muninn_core::Metric;

use crate::scalar;

/// An instruction-set family with its own set of kernels.
///
/// The derived order (`Scalar < Neon < Avx2 < Avx512`) is the ladder used
/// by `MUNINN_ISA`: requesting an instruction set selects the best one the
/// CPU supports that is not above the request. So `avx512` on an AVX2-only
/// machine gives `avx2`, `neon` on x86-64 gives `scalar`, and `avx2` on
/// AArch64 gives `neon`. Unknown values are ignored.
///
/// Requirements per family:
/// - `Avx512`: AVX-512 F and BW (plus AVX2, FMA, F16C, POPCNT). AVX-512
///   VNNI and VPOPCNTDQ are used for the integer kernels when present.
/// - `Avx2`: AVX2, FMA, F16C and POPCNT.
/// - `Neon`: AArch64 Advanced SIMD; the dot-product extension is used for
///   the integer kernels when present.
/// - `Scalar`: always available.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Isa {
  /// Portable Rust, see [`crate::scalar`].
  Scalar,
  /// AArch64 Advanced SIMD.
  Neon,
  /// x86-64 AVX2 with FMA and F16C.
  Avx2,
  /// x86-64 AVX-512 (F, BW; VNNI and VPOPCNTDQ when present).
  Avx512,
}

impl Isa {
  /// Every instruction set, lowest first.
  pub const ALL: [Isa; 4] = [Isa::Scalar, Isa::Neon, Isa::Avx2, Isa::Avx512];

  /// Short lowercase name, as accepted by `MUNINN_ISA`.
  pub fn name(self) -> &'static str {
    match self {
      Isa::Scalar => "scalar",
      Isa::Neon => "neon",
      Isa::Avx2 => "avx2",
      Isa::Avx512 => "avx512",
    }
  }

  /// Parses a name as accepted by `MUNINN_ISA` (case-insensitive).
  pub fn parse(name: &str) -> Option<Isa> {
    let name = name.trim();
    Isa::ALL
      .into_iter()
      .find(|isa| isa.name().eq_ignore_ascii_case(name))
  }

  /// Whether the running CPU supports this instruction set.
  pub fn is_supported(self) -> bool {
    kernels_for(self).is_some()
  }

  /// The best instruction set the running CPU supports, ignoring
  /// `MUNINN_ISA`.
  pub fn best() -> Isa {
    select(None, Isa::is_supported)
  }
}

/// Signature of [`Kernels::distance_gather`]: metric, query, rows, ids,
/// output.
pub type GatherFn = fn(Metric, &[f32], &[f32], &[u32], &mut [f32]);

/// Signature of [`Kernels::dot_f32_u8_gather`]: weights, codes, ids,
/// output.
pub type GatherU8Fn = fn(&[f32], &[u8], &[u32], &mut [f32]);

/// Function pointers to one instruction set's kernels.
///
/// Every field has the contract of the free function of the same name in
/// the crate root. Obtain a table with [`kernels`] (the dispatched one) or
/// [`kernels_for`]; copying a pointer out of it and calling it in a hot
/// loop avoids the per-call dispatch lookup.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Kernels {
  /// The instruction set these kernels use.
  pub isa: Isa,
  /// The CPU features the kernels actually use, e.g.
  /// `"avx512f+bw+vnni+vpopcntdq"`, for benchmark reports.
  pub features: &'static str,
  /// See [`crate::dot`].
  pub dot: fn(&[f32], &[f32]) -> f32,
  /// See [`crate::l2sq`].
  pub l2sq: fn(&[f32], &[f32]) -> f32,
  /// See [`crate::distance_batch`].
  pub distance_batch: fn(Metric, &[f32], &[f32], &mut [f32]),
  /// See [`crate::distance_gather`], which checks its arguments; called
  /// directly, an id that does not name a whole row scores NaN, and ids
  /// beyond `out` are ignored.
  pub distance_gather: GatherFn,
  /// See [`crate::dot_f32_u8`].
  pub dot_f32_u8: fn(&[f32], &[u8]) -> f32,
  /// See [`crate::dot_f32_u8_gather`], which checks its arguments; called
  /// directly, an id that does not name a whole row scores NaN, and ids
  /// beyond `out` are ignored.
  pub dot_f32_u8_gather: GatherU8Fn,
  /// See [`crate::dot_i8`].
  pub dot_i8: fn(&[i8], &[i8]) -> i32,
  /// See [`crate::l2sq_u8`].
  pub l2sq_u8: fn(&[u8], &[u8]) -> u32,
  /// See [`crate::hamming`].
  pub hamming: fn(&[u64], &[u64]) -> u32,
  /// See [`crate::dot_f16`].
  pub dot_f16: fn(&[u16], &[u16]) -> f32,
  /// See [`crate::l2sq_f16`].
  pub l2sq_f16: fn(&[u16], &[u16]) -> f32,
  /// See [`crate::f16_to_f32_slice`].
  pub f16_to_f32: fn(&[u16], &mut [f32]),
  /// See [`crate::f32_to_f16_slice`].
  pub f32_to_f16: fn(&[f32], &mut [u16]),
}

pub(crate) static SCALAR: Kernels = Kernels {
  isa: Isa::Scalar,
  features: "scalar",
  dot: scalar::dot,
  l2sq: scalar::l2sq,
  distance_batch: scalar::distance_batch,
  distance_gather: scalar::distance_gather,
  dot_f32_u8: scalar::dot_f32_u8,
  dot_f32_u8_gather: scalar::dot_f32_u8_gather,
  dot_i8: scalar::dot_i8,
  l2sq_u8: scalar::l2sq_u8,
  hamming: scalar::hamming,
  dot_f16: scalar::dot_f16,
  l2sq_f16: scalar::l2sq_f16,
  f16_to_f32: scalar::f16_to_f32_slice,
  f32_to_f16: scalar::f32_to_f16_slice,
};

/// The kernels for `isa`, or `None` if the running CPU (or the target
/// architecture) does not support it. Useful for benchmarks and tests that
/// compare instruction sets; normal code calls [`kernels`].
pub fn kernels_for(isa: Isa) -> Option<&'static Kernels> {
  match isa {
    Isa::Scalar => Some(&SCALAR),
    #[cfg(target_arch = "x86_64")]
    Isa::Avx2 => crate::avx2::kernels(),
    #[cfg(target_arch = "x86_64")]
    Isa::Avx512 => crate::avx512::kernels(),
    #[cfg(target_arch = "aarch64")]
    Isa::Neon => crate::neon::kernels(),
    #[allow(unreachable_patterns)]
    _ => None,
  }
}

/// The kernels every free function of this crate dispatches to: the best
/// instruction set the CPU supports, capped by `MUNINN_ISA`. Resolved once
/// on first use.
#[inline]
pub fn kernels() -> &'static Kernels {
  static ACTIVE: OnceLock<&'static Kernels> = OnceLock::new();
  ACTIVE.get_or_init(|| {
    let requested = std::env::var("MUNINN_ISA").ok();
    let requested = requested.as_deref().and_then(Isa::parse);
    let isa = select(requested, Isa::is_supported);
    // `select` only returns supported instruction sets, and scalar is
    // always supported.
    kernels_for(isa).unwrap_or(&SCALAR)
  })
}

/// The highest instruction set that is supported and not above
/// `requested`. Scalar is always supported.
pub(crate) fn select(
  requested: Option<Isa>,
  supported: impl Fn(Isa) -> bool,
) -> Isa {
  Isa::ALL
    .into_iter()
    .rev()
    .filter(|&isa| requested.is_none_or(|cap| isa <= cap))
    .find(|&isa| isa == Isa::Scalar || supported(isa))
    .unwrap_or(Isa::Scalar)
}
