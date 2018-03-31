#[macro_use]
extern crate criterion;
extern crate radix_tree;
extern crate rand;

use criterion::{Bencher, Criterion, Fun};
use rand::Rng;
use std::collections::btree_set::BTreeSet;
use std::collections::HashSet;
use std::hash::Hash;

use radix_tree::{ARTSet, Digital};

/// Barebones set trait to abstract over various collections.
trait Set<T> {
    fn new() -> Self;
    fn contains(&self, t: &T) -> bool;
    fn insert(&mut self, t: T);
    fn delete(&mut self, t: &T) -> bool;
}

impl<T: for<'a> Digital<'a> + Ord> Set<T> for ARTSet<T> {
    fn new() -> Self {
        ARTSet::new()
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
    let mut rng = rand::thread_rng();
    (0..len.next_power_of_two())
        .map(|_| rng.gen_range::<u64>(0, max_val))
        .collect()
}

fn random_string_vec(max_len: usize, len: usize) -> Vec<String> {
    let mut rng = rand::thread_rng();
    (0..len.next_power_of_two())
        .map(|_| {
            let s_len = rng.gen_range::<usize>(0, max_len);
            String::from_utf8((0..s_len).map(|_| rng.gen_range::<u8>(0, 128)).collect()).unwrap()
        })
        .collect()
}

fn bench_set_rand_int_lookup<S: Set<u64>>(b: &mut Bencher, contents: &S, lookups: &Vec<u64>) {
    assert!(lookups.len().is_power_of_two());
    let mut ix = 0;
    b.iter(|| {
        contents.contains(&lookups[ix]);
        ix += 1;
        ix = ix & (lookups.len() - 1);
    })
}

fn bench_set_rand_int_lookup_in_set<S: Set<u64>>(
    b: &mut Bencher,
    mut set_size: usize,
    max_elt: u64,
) {
    set_size = set_size.next_power_of_two();
    let mut s = S::new();
    let mut rng = rand::thread_rng();
    let elts: Vec<u64> = (0..set_size)
        .map(|_| rng.gen_range::<u64>(0, max_elt))
        .collect();
    for i in elts.iter() {
        s.insert(*i);
    }
    let mut ix = 0;
    b.iter(|| {
        s.contains(&elts[ix]);
        ix += 1;
        ix = ix & (set_size - 1);
    })
}

fn bench_set_rand_string<S: Set<String>>(b: &mut Bencher, mut set_size: usize, max_len: usize) {
    let mut rng = rand::thread_rng();
    set_size = set_size.next_power_of_two();
    let elts: Vec<String> = (0..set_size)
        .map(|_| {
            let s_len = rng.gen_range::<usize>(0, max_len);
            String::from_utf8((0..s_len).map(|_| rng.gen_range::<u8>(0, 128)).collect()).unwrap()
        })
        .collect();
    let mut s = S::new();
    for i in elts.iter() {
        s.insert(i.clone());
    }
    let mut ix = 0;
    b.iter(|| {
        s.contains(&elts[ix]);
        ix += 1;
        ix = ix & (set_size - 1);
    })
}

fn bench_set_rand_int_lookup_not_in_set<S: Set<u64>>(
    b: &mut Bencher,
    mut set_size: usize,
    max_elt: u64,
) {
    set_size = set_size.next_power_of_two();
    let mut s = S::new();
    let mut rng = rand::thread_rng();
    let elts: Vec<u64> = (0..set_size)
        .map(|_| rng.gen_range::<u64>(0, max_elt))
        .collect();
    let elts2: Vec<u64> = (0..set_size).map(|_| rng.gen_range::<u64>(0, !0)).collect();
    for i in elts.iter() {
        s.insert(*i);
    }
    let mut ix = 0;
    b.iter(|| {
        s.contains(&elts2[ix]);
        ix += 1;
        ix = ix & (set_size - 1);
    })
}

fn bench_set_rand_int_insert_remove<S: Set<u64>>(
    b: &mut Bencher,
    mut set_size: usize,
    max_elt: u64,
) {
    set_size = set_size.next_power_of_two();
    let mut s = S::new();
    let mut rng = rand::thread_rng();
    let elts: Vec<u64> = (0..set_size)
        .map(|_| rng.gen_range::<u64>(0, max_elt))
        .collect();
    for i in elts.iter() {
        s.insert(*i);
    }
    let mut ix = 0;
    b.iter(|| {
        s.insert(elts[ix]);
        s.delete(&elts[(ix + 2) & (set_size - 1)]);
        ix += 1;
        ix = ix & (set_size - 1);
    })
}

fn criterion_benchmark(c: &mut Criterion) {
    use std::fmt::{Debug, Error, Formatter};
    #[derive(Clone)]
    struct SizeVec(Vec<u64>, Vec<u64>);
    impl Debug for SizeVec {
        fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
            write!(f, "{:?}", self.0.len())
        }
    }
    let v1s: Vec<SizeVec> = [16 << 10, 1 << 20, 16 << 20]
        .iter()
        .map(|size: &usize| SizeVec(random_vec(*size, !0), random_vec(*size, !0)))
        .collect();

    fn make_fns<S: Set<u64>>() -> Vec<Fun<()>> {
        vec![
            // Fun::new("16k_lookup_hit", |b, _| {
            //     bench_set_rand_int_lookup_in_set::<S>(b, 16 << 10, !0)
            // }),
            // Fun::new("1M_lookup_hit", |b, _| {
            //     bench_set_rand_int_lookup_in_set::<S>(b, 1 << 20, !0)
            // }),
            // Fun::new("16k_lookup_miss", |b, _| {
            //     bench_set_rand_int_lookup_not_in_set::<S>(b, 16 << 10, !0)
            // }),
            // Fun::new("1M_lookup_miss", |b, _| {
            //     bench_set_rand_int_lookup_not_in_set::<S>(b, 1 << 20, !0)
            // }),
            Fun::new("16k_insert_remove", |b, _| {
                bench_set_rand_int_insert_remove::<S>(b, 16 << 10, !0)
            }),
            Fun::new("1M_insert_remove", |b, _| {
                bench_set_rand_int_insert_remove::<S>(b, 1 << 20, !0)
            }),
        ]
    }
    fn make_bench<S: Set<u64> + 'static>(c: &mut Criterion, desc: &str, inp: &Vec<SizeVec>) {
        eprintln!("Generating for {} (1/2)", desc);
        struct Wrap<T>(SizeVec, Box<T>);
        impl<T> Debug for Wrap<T> {
            fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
                write!(f, "{:?}", self.0)
            }
        }
        let sets1 = inp.iter()
            .map(|sv| {
                let mut s = S::new();
                for i in sv.0.iter() {
                    s.insert(*i);
                }
                Wrap(sv.clone(), Box::new(s))
            })
            .collect::<Vec<Wrap<_>>>();
        c.bench_function_over_inputs(
            &format!("{}/lookup_hit", desc),
            |b, &Wrap(ref sv, ref s)| bench_set_rand_int_lookup::<S>(b, &*s, &sv.0),
            sets1,
        );
        eprintln!("Generating for {} (2/2)", desc);
        let sets2 = inp.iter()
            .map(|sv| {
                let mut s = S::new();
                for i in sv.0.iter() {
                    s.insert(*i);
                }
                Wrap(sv.clone(), Box::new(s))
            })
            .collect::<Vec<Wrap<_>>>();
        c.bench_function_over_inputs(
            &format!("{}/lookup_miss", desc),
            |b, &Wrap(ref sv, ref s)| bench_set_rand_int_lookup::<S>(b, &*s, &sv.1),
            sets2,
        );
        c.bench_functions(desc, make_fns::<S>(), ());
    }
    make_bench::<HashSet<u64>>(c, "Hashtable", &v1s);
    make_bench::<BTreeSet<u64>>(c, "BTree", &v1s);
    make_bench::<ARTSet<u64>>(c, "ART", &v1s);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
