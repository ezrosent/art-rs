#[macro_use]
extern crate criterion;
extern crate radix_tree;
extern crate rand;

use criterion::{Bencher, Criterion};
use rand::{Rng, SeedableRng, StdRng};
use std::collections::btree_set::BTreeSet;
use std::collections::HashSet;
use std::hash::Hash;

use radix_tree::{ARTSet, ArtElement, Digital, CachingARTSet, PrefixCache, RawART};

const RAND_SEED: [usize; 32] = [1; 32];

/// Barebones set trait to abstract over various collections.
trait Set<T> {
    fn new() -> Self;
    fn contains(&self, t: &T) -> bool;
    fn insert(&mut self, t: T);
    fn delete(&mut self, t: &T) -> bool;
}

impl<T: for<'a> Digital<'a> + Ord, C: PrefixCache<ArtElement<T>>> Set<T>
    for RawART<ArtElement<T>, C>
{
    fn new() -> Self {
        Self::new()
    }
    fn contains(&self, t: &T) -> bool {
        self.contains(t)
    }
    fn insert(&mut self, t: T) {
        self.replace(t);
    }
    fn delete(&mut self, t: &T) -> bool {
        self.remove(t)
    }
}

impl<T: Hash + Eq> Set<T> for HashSet<T> {
    fn new() -> Self {
        HashSet::new()
    }
    fn contains(&self, t: &T) -> bool {
        self.get(t).is_some()
    }
    fn insert(&mut self, t: T) {
        self.replace(t);
    }
    fn delete(&mut self, t: &T) -> bool {
        self.remove(t)
    }
}

impl<T: Ord> Set<T> for BTreeSet<T> {
    fn new() -> Self {
        BTreeSet::new()
    }
    fn contains(&self, t: &T) -> bool {
        self.get(t).is_some()
    }
    fn insert(&mut self, t: T) {
        self.replace(t);
    }
    fn delete(&mut self, t: &T) -> bool {
        self.remove(t)
    }
}

fn random_vec(len: usize, max_val: u64) -> Vec<u64> {
    let mut rng = StdRng::from_seed(&RAND_SEED[..]);
    (0..len.next_power_of_two())
        .map(|_| rng.gen_range::<u64>(0, max_val))
        .collect()
}

fn random_string_vec(max_len: usize, len: usize) -> Vec<String> {
    let mut rng = StdRng::from_seed(&RAND_SEED[..]);
    (0..len.next_power_of_two())
        .map(|_| {
            let s_len = rng.gen_range::<usize>(0, max_len);
            String::from_utf8((0..s_len).map(|_| rng.gen_range::<u8>(0, 128)).collect()).unwrap()
        })
        .collect()
}

fn bench_set_rand_int_lookup<T: for <'a> Digital<'a>, S: Set<T>>(b: &mut Bencher, contents: &S, lookups: &Vec<T>) {
    assert!(lookups.len().is_power_of_two());
    let mut ix = 0;
    b.iter(|| {
        contents.contains(&lookups[ix]);
        ix += 1;
        ix = ix & (lookups.len() - 1);
    })
}

fn bench_set_insert_remove<T: Clone + for<'a> Digital<'a>, S: Set<T>>(b: &mut Bencher, contents: &mut S, lookups: &Vec<T>) {
    assert!(lookups.len().is_power_of_two());
    let mut ix = 0;
    b.iter(|| {
        contents.insert(lookups[ix].clone());
        ix += 1;
        ix = ix & (lookups.len() - 1);
        contents.delete(&lookups[ix]);
        ix += 1;
        ix = ix & (lookups.len() - 1);
    })
}

fn criterion_benchmark(c: &mut Criterion) {
    use std::fmt::{Debug, Error, Formatter};
    #[derive(Clone)]
    struct SizeVec<T>(Vec<T>, Vec<T>);
    impl<T> Debug for SizeVec<T> {
        fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
            write!(f, "{:?}", self.0.len())
        }
    }
    eprintln!("Generating Ints");
    let v1s: Vec<SizeVec<u64>> = [16 << 10, 16 << 20, 128 << 20]
        .iter()
        .map(|size: &usize| SizeVec(random_vec(*size, !0), random_vec(*size, !0)))
        .collect();

    eprintln!("Generating Strings");
    let v2s: Vec<SizeVec<String>> = [16 << 10, 64 << 20]
        .iter()
        .map(|size: &usize| SizeVec(random_string_vec(30, *size), random_string_vec(30, *size)))
        .collect();

    fn make_bench<T: 'static + Clone + for<'a> Digital<'a>, S: Set<T> + 'static>(c: &mut Criterion, desc: String, inp: &Vec<SizeVec<T>>) {
        eprintln!("Generating for {} (1/3)", desc);
        struct Wrap<S, T>(SizeVec<S>, Box<T>);
        impl<S, T> Debug for Wrap<S, T> {
            fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
                write!(f, "{:?}", self.0)
            }
        }
        let sets1 = inp.iter()
            .map(|sv| {
                let mut s = S::new();
                for i in sv.0.iter() {
                    s.insert(i.clone());
                }
                Wrap(sv.clone(), Box::new(s))
            })
            .collect::<Vec<Wrap<_,_>>>();
        c.bench_function_over_inputs(
            &format!("{}/lookup_hit", desc),
            |b, &Wrap(ref sv, ref s)| bench_set_rand_int_lookup::<T, S>(b, &*s, &sv.0),
            sets1,
        );
        eprintln!("Generating for {} (2/3)", desc);
        let sets2 = inp.iter()
            .map(|sv| {
                let mut s = S::new();
                for i in sv.0.iter() {
                    s.insert(i.clone());
                }
                Wrap(sv.clone(), Box::new(s))
            })
            .collect::<Vec<Wrap<_,_>>>();
        c.bench_function_over_inputs(
            &format!("{}/lookup_miss", desc),
            |b, &Wrap(ref sv, ref s)| bench_set_rand_int_lookup::<T, S>(b, &*s, &sv.1),
            sets2,
        );
        eprintln!("Generating for {} (3/3)", desc);
        use std::cell::UnsafeCell;
        let sets3 = inp.iter()
            .map(|sv| {
                let mut s = S::new();
                for i in sv.0.iter() {
                    s.insert(i.clone());
                }
                Wrap(sv.clone(), Box::new(UnsafeCell::new(s)))
            })
            .collect::<Vec<Wrap<_,_>>>();
        unsafe {
            c.bench_function_over_inputs(
                &format!("{}/insert_remove", desc),
                |b, &Wrap(ref sv, ref s)| bench_set_insert_remove::<T, S>(b, &mut *s.get(), &sv.0),
                sets3,
            );
        }
    }
    macro_rules! bench_inner {
        ($c:expr, $container:tt, $ivec:expr, $svec:expr) => {
            {
                make_bench::<u64, $container<u64>>($c, format!("{}/u64", stringify!($container)), $ivec);
                make_bench::<String, $container<String>>($c, format!("{}/String", stringify!($container)), $svec);
            }
        };
    }
    macro_rules! bench_all {
        ($c:expr, $ivec:expr, $svec:expr, $( $container:tt ),+) => {
            $(
                bench_inner!($c, $container, $ivec, $svec);
            )+
        }
    }
    bench_all!(c, &v1s, &v2s, HashSet, BTreeSet, ARTSet, CachingARTSet);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
