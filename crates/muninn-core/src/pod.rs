/// Plain-old-data: types that can be viewed directly as bytes and back.
///
/// # Safety
///
/// Implementors must be `Copy`, have no padding bytes, no pointers or
/// references, and accept every bit pattern as a valid value.
pub unsafe trait Pod: Copy + Send + Sync + 'static {}

macro_rules! impl_pod {
  ($($t:ty),*) => {
    // SAFETY: primitive integers and floats satisfy every Pod requirement.
    $(unsafe impl Pod for $t {})*
  };
}

impl_pod!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

/// Views a slice of `Pod` values as raw bytes (native endianness).
#[inline]
pub fn as_bytes<T: Pod>(values: &[T]) -> &[u8] {
  // SAFETY: T is Pod, so every byte of the slice is initialised and the
  // byte view covers exactly the same memory with alignment 1.
  unsafe {
    std::slice::from_raw_parts(
      values.as_ptr().cast::<u8>(),
      std::mem::size_of_val(values),
    )
  }
}
