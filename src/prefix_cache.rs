extern crate fnv;
#[cfg(feature = "print_cache_stats")]
use std::cell::UnsafeCell;
use std::cmp;
use std::marker::PhantomData;
use std::ptr;

use super::art_internal::MarkedPtr;
use super::byteorder::{BigEndian, ByteOrder};

pub use self::dense_hash_set::HashSetPrefixCache;

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

mod dense_hash_set {
    use super::*;
    use super::fnv::FnvHasher;

    use std::hash::{Hash, Hasher};
    use std::mem;

    pub struct HashSetPrefixCache<T>(DenseHashTable<MarkedElt<T>>);
    impl<T> PrefixCache<T> for HashSetPrefixCache<T> {
        const ENABLED: bool = true;
        const COMPLETE: bool = false;
        fn new(_buckets: usize) -> Self {
            HashSetPrefixCache(DenseHashTable::new())
        }
        fn lookup(&self, bs: &[u8]) -> Option<MarkedPtr<T>> {
            let prefix = HashBuckets::<T>::read_u64(bs);
            self.0.lookup(&prefix).map(|elt| elt.ptr.clone())
        }

        fn insert(&mut self, bs: &[u8], ptr: MarkedPtr<T>) {
            let prefix = HashBuckets::<T>::read_u64(bs);
            if ptr.is_null() {
                let _ = self.0.delete(&prefix);
            } else {
                let _ = self.0.insert(MarkedElt {
                    prefix: prefix,
                    ptr: ptr,
                });
            }
        }
    }

    trait DHTE {
        type Key;
        fn null() -> Self;
        fn tombstone() -> Self;
        fn is_null(&self) -> bool;
        fn is_tombstone(&self) -> bool;
        fn key(&self) -> &Self::Key;
    }

    const MARKED_TOMBSTONE: usize = !0;
    struct MarkedElt<T> {
        prefix: u64,
        ptr: MarkedPtr<T>,
    }

    impl<T> DHTE for MarkedElt<T> {
        type Key = u64;
        fn null() -> Self {
            MarkedElt {
                prefix: 0,
                ptr: MarkedPtr::null(),
            }
        }
        fn tombstone() -> Self {
            MarkedElt {
                prefix: 0,
                ptr: MarkedPtr::from_leaf(MARKED_TOMBSTONE as *mut T),
            }
        }

        fn is_null(&self) -> bool {
            self.ptr.is_null()
        }
        fn is_tombstone(&self) -> bool {
            self.ptr.raw_eq(MARKED_TOMBSTONE)
        }
        fn key(&self) -> &Self::Key {
            &self.prefix
        }
    }

    /// A bare-bones implementation of Google's dense_hash_set. Not a full-featured map, but
    /// contains sufficient functionality to be used as a PrefixCache
    struct DenseHashTable<T> {
        buckets: Vec<T>,
        len: usize,
        set: usize,
    }

    impl<T: DHTE> DenseHashTable<T>
    where
        T::Key: Eq + Hash,
    {
        fn new() -> Self {
            DenseHashTable {
                buckets: Vec::new(),
                len: 0,
                set: 0,
            }
        }

        fn grow(&mut self) {
            if self.buckets.len() == 0 {
                self.buckets.push(T::null());
                return;
            } else {
                let l = self.buckets.len();
                self.buckets.extend((0..l).map(|_| T::null()));
            }
            debug_assert!(self.buckets.len().is_power_of_two());
            let mut v = Vec::with_capacity(self.len);
            let l = self.buckets.len();
            for ix in 0..(l / 2) {
                let i = unsafe { self.buckets.get_unchecked_mut(ix) };
                if i.is_null() {
                    continue;
                }
                if i.is_tombstone() {
                    *i = T::null();
                    continue;
                }
                let mut t = T::null();
                mem::swap(i, &mut t);
                v.push(t);
            }
            for elt in v.into_iter() {
                let _res = self.insert(elt);
                debug_assert!(_res.is_ok());
            }
        }

        fn lookup(&self, k: &T::Key) -> Option<&T> {
            if self.buckets.len() == 0 {
                return None;
            }
            let hash = {
                let mut hasher = FnvHasher::default();
                k.hash(&mut hasher);
                (hasher.finish() & (self.buckets.len() as u64 - 1)) as usize
            };
            let mut ix = hash;
            let mut times = 0;
            while times < self.buckets.len() {
                debug_assert!(ix < self.buckets.len());
                times += 1;
                let bucket = unsafe { self.buckets.get_unchecked(ix) };
                if bucket.is_null() {
                    return None;
                }
                if bucket.is_tombstone() || bucket.key() != k {
                    ix = hash + times * times;
                    ix &= self.buckets.len() - 1;
                    continue;
                }
                return Some(bucket);
            }
            return None;
        }

        fn delete(&mut self, k: &T::Key) -> Option<T> {
            let hash = {
                let mut hasher = FnvHasher::default();
                k.hash(&mut hasher);
                (hasher.finish() & (self.buckets.len() as u64 - 1)) as usize
            };
            let mut ix = hash;
            let mut times = 0;
            let l = self.buckets.len();
            while times < l {
                debug_assert!(ix < self.buckets.len());
                times += 1;
                let bucket = unsafe { self.buckets.get_unchecked_mut(ix) };
                if bucket.is_null() {
                    return None;
                }
                if bucket.is_tombstone() || bucket.key() != k {
                    ix = hash + times * times;
                    ix &= l - 1;
                    continue;
                }
                let mut deleted = T::tombstone();
                mem::swap(bucket, &mut deleted);
                self.len -= 1;
                return Some(deleted);
            }
            return None;
        }

        fn insert(&mut self, mut t: T) -> Result<(), T> {
            if self.set >= self.buckets.len() >> 1 {
                self.grow();
            }
            debug_assert!(!t.is_null());
            debug_assert!(!t.is_tombstone());
            let hash = {
                let mut hasher = FnvHasher::default();
                t.key().hash(&mut hasher);
                (hasher.finish() & (self.buckets.len() as u64 - 1)) as usize
            };
            let mut ix = hash;
            let mut times = 0;
            let l = self.buckets.len();
            while times < l {
                debug_assert_eq!(l, self.buckets.len());
                debug_assert!(ix < self.buckets.len());
                times += 1;
                let bucket = unsafe { self.buckets.get_unchecked_mut(ix) };
                if bucket.is_null() || bucket.is_tombstone() {
                    mem::swap(bucket, &mut t);
                    self.len += 1;
                    self.set += 1;
                    return Ok(());
                }
                if bucket.key() == t.key() {
                    mem::swap(bucket, &mut t);
                    return Err(t);
                }

                ix = hash + times * times;
                ix &= l - 1;
            }
            panic!(
                "table too small! blen={}, set={}, len={}, l={}",
                self.buckets.len(),
                self.set,
                self.len,
                l
            )
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use super::super::super::rand;
        use super::super::super::rand::Rng;
        fn random_vec(max_val: usize, len: usize) -> Vec<usize> {
            let mut rng = rand::thread_rng();
            (0..len)
                .map(|_| rng.gen_range::<usize>(0, max_val))
                .collect()
        }

        #[derive(Debug)]
        struct UsizeElt(usize, usize);
        impl DHTE for UsizeElt {
            type Key = usize;
            fn null() -> Self {
                UsizeElt(0, 0)
            }
            fn tombstone() -> Self {
                UsizeElt(0, 2)
            }
            fn is_null(&self) -> bool {
                self.1 == 0
            }
            fn is_tombstone(&self) -> bool {
                self.1 == 2
            }
            fn key(&self) -> &Self::Key {
                &self.0
            }
        }

        impl UsizeElt {
            fn new(u: usize) -> Self {
                UsizeElt(u, 1)
            }
        }

        #[test]
        fn dense_hash_set_smoke_test() {
            let mut s = DenseHashTable::<UsizeElt>::new();
            let mut v1 = random_vec(!0, 1 << 18);
            for item in v1.iter() {
                let _ = s.insert(UsizeElt::new(*item));
                assert!(
                    s.lookup(item).is_some(),
                    "lookup failed immediately for {:?}",
                    *item
                );
            }
            let mut missing = Vec::new();
            for item in v1.iter() {
                if s.lookup(item).is_none() {
                    missing.push(*item)
                }
            }
            assert_eq!(missing.len(), 0, "missing={:?}", missing);
            v1.sort();
            v1.dedup_by_key(|x| *x);
            let mut v2 = Vec::new();
            for _ in 0..(1 << 17) {
                if let Some(x) = v1.pop() {
                    v2.push(x)
                } else {
                    break;
                }
            }
            let mut failures = 0;
            for i in v2.iter() {
                let mut fail = 0;
                if s.lookup(i).is_none() {
                    eprintln!("{:?} no longer in the set!", *i);
                    fail = 1;
                }
                let res = s.delete(i);
                if res.is_none() {
                    fail = 1;
                }
                if s.lookup(i).is_some() {
                    fail = 1;
                }
                failures += fail;
            }
            assert_eq!(failures, 0);
            let mut failed = false;
            for i in v2.iter() {
                if s.lookup(i).is_some() {
                    eprintln!("Deleted {:?}, but it's still there!", *i);
                    failed = true;
                };
            }
            assert!(!failed);
            for i in v1.iter() {
                assert!(
                    s.lookup(i).is_some(),
                    "Didn't delete {:?}, but it is gone!",
                    *i
                );
            }
        }
    }
}
