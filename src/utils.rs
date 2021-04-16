use std::cell::UnsafeCell;
use std::slice::{self};

/// A slice type which can be shared between threads, but must be fully managed by the caller.
/// Any synchronization must be ensured by the caller, which is why all access is `unsafe`.
#[derive(Debug)]
pub struct UnsafeSlice<'a, T> {
    // holds the data to ensure lifetime correctness
    data: UnsafeCell<&'a mut [T]>,
    /// pointer to the data
    ptr: *mut T,
    /// Number of elements, not bytes.
    len: usize,
}

unsafe impl<'a, T> Sync for UnsafeSlice<'a, T> {}
unsafe impl<'a, T> Send for UnsafeSlice<'a, T> {}

impl<'a, T> UnsafeSlice<'a, T> {
    /// Takes mutable slice, to ensure that `UnsafeSlice` is the only user of this memory, until it gets dropped.
    pub fn from_slice(source: &'a mut [T]) -> Self {
        let len = source.len();
        let ptr = source.as_mut_ptr();
        let data = UnsafeCell::new(source);
        Self { data, ptr, len }
    }

    /// Safety: The caller must ensure that there are no unsynchronized parallel access to the same regions.
    #[inline]
    pub unsafe fn as_mut_slice(&self) -> &'a mut [T] {
        slice::from_raw_parts_mut(self.ptr, self.len)
    }
    /// Safety: The caller must ensure that there are no unsynchronized parallel access to the same regions.
    #[inline]
    pub unsafe fn as_slice(&self) -> &'a [T] {
        slice::from_raw_parts(self.ptr, self.len)
    }

    #[inline]
    pub unsafe fn get(&self, index: usize) -> &'a T {
        &*self.ptr.add(index)
    }

    #[inline]
    pub unsafe fn get_mut(&self, index: usize) -> &'a mut T {
        &mut *self.ptr.add(index)
    }
}
