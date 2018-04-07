extern crate fnv;
#[cfg(feature="print_cache_stats")]
use std::cell::UnsafeCell;
use std::cmp;
use std::marker::PhantomData;
use std::ptr;

use super::art_internal::MarkedPtr;
use super::byteorder::{BigEndian, ByteOrder};

/// PrefixCache describes types that can cache pointers interior to an ART.
pub trait PrefixCache<T> {
    /// If true, the cache is used during ART set operations. If false, the cache is ignored.
    const ENABLED: bool;
    /// If true, lookup returning None indicates that no nodes with prefix `bs` are in the set.
    const COMPLETE: bool;
    fn new(buckets: usize) -> Self;
    fn lookup(&self, bs: &[u8]) -> Option<MarkedPtr<T>>;
    fn insert(&mut self, bs: &[u8], ptr: MarkedPtr<T>);
}
pub struct NullBuckets<T>(PhantomData<T>);

impl<T> PrefixCache<T> for NullBuckets<T> {
    const ENABLED: bool = false;
    const COMPLETE: bool = false;
    fn new(_: usize) -> Self {
        NullBuckets(PhantomData)
    }
    fn lookup(&self, _: &[u8]) -> Option<MarkedPtr<T>> {
        None
    }
    fn insert(&mut self, _: &[u8], _ptr: MarkedPtr<T>) {}
}

pub struct HashBuckets<T> {
    data: Vec<(u64, MarkedPtr<T>)>,
    len: usize,
    #[cfg(feature = "print_cache_stats")]
    misses: UnsafeCell<usize>,
    #[cfg(feature = "print_cache_stats")]
    hits: UnsafeCell<usize>,
    #[cfg(feature = "print_cache_stats")]
    collisions: UnsafeCell<usize>,
    #[cfg(feature = "print_cache_stats")]
    overwrites: usize,
}

impl<T> Drop for HashBuckets<T> {
    fn drop(&mut self) {
        #[cfg(feature = "print_cache_stats")]
        unsafe {
            let h = *self.hits.get();
            let m = *self.misses.get();
            let c = *self.collisions.get();
            eprintln!(
                "hits={:?} miss={:?} collisions={:?} hit rate={:?} len={:?} overwrites={:?}",
                h,
                m,
                c,
                h as f64 / (h + m + c) as f64,
                self.len,
                self.overwrites
            );
        }
    }
}

impl<T> PrefixCache<T> for HashBuckets<T> {
    const ENABLED: bool = true;
    const COMPLETE: bool = false;

    fn new(size: usize) -> Self {
        #[cfg(feature = "print_cache_stats")]
        {
            HashBuckets {
                data: (0..size.next_power_of_two())
                    .map(|_| (0, MarkedPtr::null()))
                    .collect::<Vec<_>>(),
                misses: UnsafeCell::new(0),
                hits: UnsafeCell::new(0),
                collisions: UnsafeCell::new(0),
                overwrites: 0,
                len: 0,
            }
        }
        #[cfg(not(feature = "print_cache_stats"))]
        {
            HashBuckets {
                data: (0..size.next_power_of_two())
                    .map(|_| (0, MarkedPtr::null()))
                    .collect::<Vec<_>>(),
                len: 0,
            }
        }
    }

    fn lookup(&self, bs: &[u8]) -> Option<MarkedPtr<T>> {
        let key = self.get_index(bs);
        let (i, ptr) = unsafe { self.data.get_unchecked(key) }.clone();
        if ptr.is_null() {
            #[cfg(feature = "print_cache_stats")]
            unsafe { *self.misses.get() += 1 };
            None
        } else {
            let key = Self::read_u64(bs);
            if key == i {
                #[cfg(feature = "print_cache_stats")]
                unsafe { *self.hits.get() += 1 };
                Some(ptr)
            } else {
                #[cfg(feature = "print_cache_stats")]
                unsafe { *self.collisions.get() += 1 };
                None
            }
        }
    }

    fn insert(&mut self, bs: &[u8], ptr: MarkedPtr<T>) {
        let h = self.get_index(bs);
        let key = Self::read_u64(bs);
        if unsafe { self.data.get_unchecked(h).1.is_null() } {
            self.len += 1;
        }
        #[cfg(feature = "print_cache_stats")]
        {
            let (k, _ptr) = unsafe { self.data.get_unchecked(h) }.clone();
            if !_ptr.is_null() && k == key {
                self.overwrites += 1;
            }
        }
        unsafe { *self.data.get_unchecked_mut(h) = (key, ptr) };
    }
}
impl<T> HashBuckets<T> {
    fn read_u64(bs: &[u8]) -> u64 {
        debug_assert!(bs.len() <= 8);
        let mut arr = [0 as u8; 8];
        unsafe { ptr::copy_nonoverlapping(&bs[0], &mut arr[0], cmp::min(bs.len(), 8)) };
        BigEndian::read_u64(&arr[..])
    }

    fn get_index(&self, bs: &[u8]) -> usize {
        debug_assert!(self.data.len().is_power_of_two());
        use self::fnv::FnvHasher;
        use std::hash::Hasher;
        let mut hasher = FnvHasher::default();
        let u = Self::read_u64(bs);
        hasher.write_u64(u);
        hasher.finish() as usize & (self.data.len() - 1)
    }
}
